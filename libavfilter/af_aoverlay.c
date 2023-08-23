/*
 * Copyright (c) 2023 Harshit Karwal
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "libavutil/opt.h"
#include "libavutil/log.h"

#include "audio.h"
#include "avfilter.h"
#include "filters.h"
#include "internal.h"
#include "formats.h"

typedef struct FrameRingBuffer {
    uint8_t *extended_data_buffer;
    int capacity;
    int start;
    int end;
    int size;
} FrameRingBuffer;

static int ring_init(FrameRingBuffer **ring, unsigned int capacity, int size)
{
    *ring = av_malloc(sizeof(FrameRingBuffer));
    if (!*ring)
        return AVERROR(ENOMEM);

    (*ring)->extended_data_buffer = av_malloc_array(capacity, size);

    if (!(*ring)->extended_data_buffer)
        return AVERROR(ENOMEM);

    (*ring)->capacity = capacity;
    (*ring)->start = 0;
    (*ring)->end = 0;
    (*ring)->size = 0;

    return 0;
}

static int ring_empty(FrameRingBuffer *ring)
{
    return ring->size == 0;
}

static int ring_full(FrameRingBuffer *ring)
{
    return ring->size == ring->capacity;
}

static int ring_insert(FrameRingBuffer **sample_buffers, AVFrame *frame, AVFilterLink *inlink)
{
    uint8_t *dst;

    for (int c = 0; c < inlink->ch_layout.nb_channels; c++) {
        for (int i = 0; i < frame->nb_samples; i++) {
            if (ring_full(sample_buffers[c]))
                return AVERROR(EPERM);

            dst = sample_buffers[c]->extended_data_buffer +
                  sample_buffers[c]->end * av_get_bytes_per_sample(inlink->format);

            sample_buffers[c]->end = (sample_buffers[c]->end + 1) % sample_buffers[c]->capacity;
            sample_buffers[c]->size++;

            memcpy(dst, frame->extended_data[c] + i * av_get_bytes_per_sample(inlink->format),
                   av_get_bytes_per_sample(inlink->format));
        }
    }

    return 0;
}

static int ring_remove(FrameRingBuffer **sample_buffers, AVFilterLink *inlink, uint8_t **dest, int nb_samples)
{
    uint8_t *src;

    for (int c = 0; c < inlink->ch_layout.nb_channels; c++) {
        for (int i = 0; i < nb_samples; i++) {
            if (ring_empty(sample_buffers[c]))
                return AVERROR(EPERM);

            src = sample_buffers[c]->extended_data_buffer +
                  sample_buffers[c]->start * av_get_bytes_per_sample(inlink->format);

            sample_buffers[c]->start = (sample_buffers[c]->start + 1) % sample_buffers[c]->capacity;
            sample_buffers[c]->size--;

            memcpy(dest[c] + i * av_get_bytes_per_sample(inlink->format), src,
                   av_get_bytes_per_sample(inlink->format));
        }
    }

    return 0;
}

static void ring_free(FrameRingBuffer *ring)
{
    av_freep(&ring->extended_data_buffer);
    av_freep(&ring);
}

typedef struct AOverlayContext {
    const AVClass *class;
    AVFrame *main_input;
    AVFrame *overlay_input;
    int64_t pts;
    int main_eof;
    int overlay_eof;

    int default_mode;
    int previous_samples;
    int64_t pts_gap;
    int64_t previous_pts;
    int64_t pts_gap_start;
    int64_t pts_gap_end;

    int is_disabled;
    int nb_channels;
    int crossfade_ready;
    FrameRingBuffer **main_sample_buffers;
    FrameRingBuffer **overlay_sample_buffers;
    int64_t cf_duration;
    int64_t cf_samples;
    void (*crossfade_samples)(uint8_t **dst, uint8_t * const *cf0,
                              uint8_t * const *cf1,
                              int nb_samples, int channels);

    int64_t transition_pts;
    int64_t transition_pts2;

    uint8_t **cf0;
    uint8_t **cf1;
} AOverlayContext;

#define SEGMENT_SIZE 1024

#define OFFSET(x) offsetof(AOverlayContext, x)

#define FLAGS AV_OPT_FLAG_AUDIO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption aoverlay_options[] = {
    { "cf_duration",    "set duration (in seconds) for cross fade between the inputs", OFFSET(cf_duration),    AV_OPT_TYPE_DURATION,   {.i64 = 100000}, 0,  60000000,   FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(aoverlay);

#define CROSSFADE_PLANAR(name, type)                                                 \
static void crossfade_samples_## name ##p(uint8_t **dst, uint8_t * const *cf0,       \
                                          uint8_t * const *cf1,                      \
                                          int nb_samples, int channels)              \
{                                                                                    \
    for (int i = 0; i < nb_samples; i++) {                                           \
        double main_gain = av_clipd(1.0 * (nb_samples - 1 - i) / nb_samples, 0, 1.); \
        double overlay_gain = av_clipd(1.0 * i / nb_samples, 0, 1.);                 \
        for (int c = 0; c < channels; c++) {                                         \
            type *d = (type *)dst[c];                                                \
            const type *s0 = (type *)cf0[c];                                         \
            const type *s1 = (type *)cf1[c];                                         \
                                                                                     \
            d[i] = s0[i] * main_gain + s1[i] * overlay_gain;                         \
        }                                                                            \
    }                                                                                \
}

CROSSFADE_PLANAR(dbl, double)
CROSSFADE_PLANAR(flt, float)
CROSSFADE_PLANAR(s16, int16_t)
CROSSFADE_PLANAR(s32, int32_t)

static av_cold int init(AVFilterContext *ctx)
{
    AOverlayContext *s  = ctx->priv;

    s->is_disabled      = 1;
    s->transition_pts   = AV_NOPTS_VALUE;
    s->transition_pts2  = AV_NOPTS_VALUE;

    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    AOverlayContext *s = ctx->priv;

    for (int i = 0; i < s->nb_channels; i++) {
        ring_free(s->main_sample_buffers[i]);
        ring_free(s->overlay_sample_buffers[i]);
        av_freep(&s->cf0[i]);
        av_freep(&s->cf1[i]);
    }
    av_freep(&s->cf0);
    av_freep(&s->cf1);

    av_freep(&s->main_sample_buffers);
    av_freep(&s->overlay_sample_buffers);

    av_frame_free(&s->main_input);
    av_frame_free(&s->overlay_input);
}

static int query_formats(AVFilterContext *ctx)
{
    int ret;

    static const enum AVSampleFormat sample_fmts[] = {
        AV_SAMPLE_FMT_DBLP, AV_SAMPLE_FMT_FLTP,
        AV_SAMPLE_FMT_S16P, AV_SAMPLE_FMT_S32P,
        AV_SAMPLE_FMT_NONE
    };

    return ff_set_common_formats_from_list(ctx, sample_fmts);
}

static int crossfade_prepare(AOverlayContext *s, AVFilterLink *main_inlink, AVFilterLink *overlay_inlink, AVFilterLink *outlink,
                             int nb_samples, AVFrame **main_buffer, AVFrame **overlay_buffer, int mode)
{
    int ret;

    *main_buffer = ff_get_audio_buffer(outlink, nb_samples);
    if (!(*main_buffer))
        return AVERROR(ENOMEM);

    (*main_buffer)->pts = s->pts;
    s->pts += av_rescale_q(nb_samples, (AVRational){ 1, outlink->sample_rate }, outlink->time_base);

    if (ret = ring_remove(s->main_sample_buffers, main_inlink, (*main_buffer)->extended_data, nb_samples) < 0)
        return ret;

    if (mode == 1) {
        s->previous_samples = (*main_buffer)->nb_samples;
    } else if (mode == -1 || (mode == 0 && s->is_disabled)) {
        *overlay_buffer = ff_get_audio_buffer(outlink, nb_samples);
        if (!(*overlay_buffer))
            return AVERROR(ENOMEM);

        if (ret = ring_remove(s->overlay_sample_buffers, overlay_inlink, (*overlay_buffer)->extended_data, nb_samples) < 0)
            return ret;

        (*overlay_buffer)->pts = (*main_buffer)->pts;
    }

    s->crossfade_ready = 1;

    return 0;
}

static int crossfade_samples(AOverlayContext *s, AVFilterLink *main_inlink, AVFilterLink *overlay_inlink, AVFilterLink *outlink,
                             int nb_samples, AVFrame **out, int mode)
{
    int ret;

    *out = ff_get_audio_buffer(outlink, nb_samples);
    if (!(*out))
        return AVERROR(ENOMEM);

    if (ret = ring_remove(s->main_sample_buffers, main_inlink, s->cf0, nb_samples) < 0)
        return ret;
    if (ret = ring_remove(s->overlay_sample_buffers, overlay_inlink, s->cf1, nb_samples) < 0)
        return ret;

    if (mode == 0) {
        s->is_disabled ? s->crossfade_samples((*out)->extended_data, s->cf1, s->cf0, nb_samples, (*out)->ch_layout.nb_channels)
                       : s->crossfade_samples((*out)->extended_data, s->cf0, s->cf1, nb_samples, (*out)->ch_layout.nb_channels);
    } else if (mode == -1) {
        s->crossfade_samples((*out)->extended_data, s->cf1, s->cf0, s->cf_samples, (*out)->ch_layout.nb_channels);
    } else if (mode == 1) {
        s->transition_pts2 != AV_NOPTS_VALUE ? s->crossfade_samples((*out)->extended_data, s->cf1, s->cf0, nb_samples, (*out)->ch_layout.nb_channels)
                                             : s->crossfade_samples((*out)->extended_data, s->cf0, s->cf1, nb_samples, (*out)->ch_layout.nb_channels);
    }

    (*out)->pts = s->pts;
    s->pts += av_rescale_q(nb_samples, (AVRational){ 1, outlink->sample_rate }, outlink->time_base);
    s->transition_pts = AV_NOPTS_VALUE;
    s->transition_pts2 = AV_NOPTS_VALUE;
    s->crossfade_ready = 0;

    return 0;
}

static int consume_samples(AOverlayContext *s, AVFilterLink *overlay_inlink, AVFilterLink *outlink)
{
    int ret, status, nb_samples;
    int64_t pts;

    nb_samples = FFMIN(SEGMENT_SIZE, s->overlay_sample_buffers[0]->capacity - s->overlay_sample_buffers[0]->size);

    ret = ff_inlink_consume_samples(overlay_inlink, nb_samples, nb_samples, &s->overlay_input);
    if (ret < 0) {
        return ret;
    } else if (ff_inlink_acknowledge_status(overlay_inlink, &status, &pts)) {
        s->overlay_eof = 1;
        return 0;
    } else if (!ret) {
        if (ff_outlink_frame_wanted(outlink))
            ff_inlink_request_frame(overlay_inlink);
        return 0;
    }

    if (ret = ring_insert(s->overlay_sample_buffers, s->overlay_input, overlay_inlink) < 0)
        return ret;

    return 1;
}

static int activate(AVFilterContext *ctx)
{
    AOverlayContext *s = ctx->priv;
    int status, ret, nb_samples;
    int64_t pts;
    AVFrame *out = NULL, *main_buffer = NULL, *overlay_buffer = NULL;

    AVFilterLink *main_inlink = ctx->inputs[0];
    AVFilterLink *overlay_inlink = ctx->inputs[1];
    AVFilterLink *outlink = ctx->outputs[0];

    FF_FILTER_FORWARD_STATUS_BACK_ALL(outlink, ctx);

    if (s->default_mode && (s->pts_gap_end - s->pts_gap_start <= 0 || s->overlay_eof)) {
        s->default_mode = 0;
        s->transition_pts2 = s->pts_gap_end;
    }

    if (s->main_sample_buffers[0]->size != s->main_sample_buffers[0]->capacity && !s->main_eof && !s->default_mode) {
        nb_samples = FFMIN(SEGMENT_SIZE, s->main_sample_buffers[0]->capacity - s->main_sample_buffers[0]->size);

        ret = ff_inlink_consume_samples(main_inlink, nb_samples, nb_samples, &s->main_input);
        if (ret > 0) {
            if (ctx->enable_str && s->is_disabled != ctx->is_disabled && !s->overlay_eof) {
                s->is_disabled = ctx->is_disabled;
                s->transition_pts = s->main_input->pts;

                if (s->main_sample_buffers[0]->size + s->main_input->nb_samples < s->main_sample_buffers[0]->capacity)
                    s->crossfade_ready = 1;
                if (s->main_sample_buffers[0]->size == 0) {
                    s->transition_pts = AV_NOPTS_VALUE;
                    s->crossfade_ready = 0;
                }
            }
            if (!ctx->enable_str && !s->default_mode) {
                if (s->previous_pts + av_rescale_q(s->previous_samples, (AVRational){ 1, outlink->sample_rate }, outlink->time_base) >= s->main_input->pts) {
                    s->default_mode = 0;
                    s->previous_pts = s->main_input->pts;
                    s->previous_samples = s->main_input->nb_samples;
                } else if (!s->overlay_eof) {
                    s->pts_gap_start = s->previous_pts;
                    if (s->pts > 0 || s->main_sample_buffers[0]->size > 0)
                        s->transition_pts = s->pts_gap_start;
                    s->pts_gap_end = s->main_input->pts;
                    s->default_mode = 1;
                }
            }

            if (ret = ring_insert(s->main_sample_buffers, s->main_input, main_inlink) < 0)
                return ret;
        } else if (ret < 0) {
            return ret;
        } else if (ff_inlink_acknowledge_status(main_inlink, &status, &pts)) {
            s->main_eof = 1;
            s->crossfade_ready = 1;
        } else if (!ret) {
            if (ff_outlink_frame_wanted(outlink))
                ff_inlink_request_frame(main_inlink);
            return 0;
        }
    }

    if (s->main_eof && s->main_sample_buffers[0]->size == 0 && ff_inlink_acknowledge_status(main_inlink, &status, &pts)) {
        ff_outlink_set_status(outlink, status, pts);
        return 0;
    }

    if (s->main_sample_buffers[0]->size < s->main_sample_buffers[0]->capacity &&
        (s->transition_pts == AV_NOPTS_VALUE || s->main_sample_buffers[0]->size != s->cf_samples) && !s->default_mode) {
        if (ff_inlink_acknowledge_status(main_inlink, &status, &pts)) {
            s->main_eof = 1;
            s->crossfade_ready = 1;
        } else {
            ff_inlink_request_frame(main_inlink);
            return 0;
        }
    }

    if (!s->overlay_eof) {
        if (s->overlay_sample_buffers[0]->size < s->overlay_sample_buffers[0]->capacity) {
            ret = consume_samples(s, overlay_inlink, outlink);
            if (ret <= 0) {
                if (!s->overlay_eof)
                    return ret;
            }
        }

        if (s->overlay_sample_buffers[0]->size < s->overlay_sample_buffers[0]->capacity) {
            if (ff_inlink_acknowledge_status(overlay_inlink, &status, &pts)) {
                s->overlay_eof = 1;
                s->transition_pts = s->pts + av_rescale_q(s->overlay_sample_buffers[0]->size - (s->cf_samples / 2),
                                                          (AVRational){ 1, outlink->sample_rate }, outlink->time_base);
                s->is_disabled = 1;
            } else {
                ff_inlink_request_frame(overlay_inlink);
                return 0;
            }
        }
    }

    if (!ctx->enable_str) {
        if (s->transition_pts != AV_NOPTS_VALUE && s->main_sample_buffers[0]->size > s->cf_samples + SEGMENT_SIZE) {
            nb_samples = s->main_sample_buffers[0]->capacity - s->cf_samples - SEGMENT_SIZE;

            if (ret = crossfade_prepare(s, main_inlink, overlay_inlink, outlink, nb_samples, &main_buffer, &overlay_buffer, 1) < 0)
                return ret;

            return ff_filter_frame(outlink, main_buffer);
        } else if (s->transition_pts != AV_NOPTS_VALUE || s->transition_pts2 != AV_NOPTS_VALUE) {
            nb_samples = FFMIN(s->cf_samples, s->main_sample_buffers[0]->size - SEGMENT_SIZE);

            if (ret = crossfade_samples(s, main_inlink, overlay_inlink, outlink, nb_samples, &out, 1) < 0)
                return ret;

            return ff_filter_frame(outlink, out);
        } else if (!s->default_mode) {
            nb_samples = FFMIN(s->main_sample_buffers[0]->size, SEGMENT_SIZE);

            main_buffer = ff_get_audio_buffer(outlink, nb_samples);
            if (!main_buffer)
                return AVERROR(ENOMEM);

            main_buffer->pts = s->pts;
            s->pts += av_rescale_q(nb_samples, (AVRational){ 1, outlink->sample_rate }, outlink->time_base);

            if (ret = ring_remove(s->main_sample_buffers, main_inlink, main_buffer->extended_data, nb_samples) < 0)
                return ret;
        }

        if (!s->default_mode || s->overlay_eof) {
            s->previous_samples = main_buffer->nb_samples;
            return ff_filter_frame(outlink, main_buffer);
        }

        s->pts_gap = s->pts_gap_end - s->pts_gap_start;

        nb_samples = FFMIN(SEGMENT_SIZE, av_rescale_q(s->pts_gap, outlink->time_base, (AVRational){ 1, outlink->sample_rate }));

        overlay_buffer = ff_get_audio_buffer(outlink, nb_samples);
        if (!overlay_buffer)
            return AVERROR(ENOMEM);

        if (ret = ring_remove(s->overlay_sample_buffers, overlay_inlink, overlay_buffer->extended_data, nb_samples) < 0)
            return ret;

        s->previous_samples = nb_samples;
        s->previous_pts += av_rescale_q(nb_samples, (AVRational){ 1, outlink->sample_rate }, outlink->time_base);
        s->pts_gap_start += av_rescale_q(nb_samples, (AVRational){ 1, outlink->sample_rate }, outlink->time_base);

        overlay_buffer->pts = s->pts;
        s->pts += av_rescale_q(nb_samples, (AVRational){ 1, outlink->sample_rate }, outlink->time_base);

        av_frame_free(&main_buffer);

        return ff_filter_frame(outlink, overlay_buffer);
    }

    if (s->overlay_eof && s->overlay_sample_buffers[0]->size > 0) {
        if (s->overlay_sample_buffers[0]->size != s->cf_samples) {
            nb_samples = s->overlay_sample_buffers[0]->size - s->cf_samples;

            if (ret = crossfade_prepare(s, main_inlink, overlay_inlink, outlink, nb_samples, &main_buffer, &overlay_buffer, -1) < 0)
                return ret;

            return ff_filter_frame(outlink, overlay_buffer);
        } else if (s->overlay_sample_buffers[0]->size == s->cf_samples) {
            if (ret = crossfade_samples(s, main_inlink, overlay_inlink, outlink, s->cf_samples, &out, -1) < 0)
                return ret;

            return ff_filter_frame(outlink, out);
        }
    }

    if (s->transition_pts != AV_NOPTS_VALUE && !s->crossfade_ready) {
        nb_samples = av_rescale_q(s->transition_pts - (s->cf_samples / 2) - s->pts, outlink->time_base, (AVRational) { 1, outlink->sample_rate });

        if (ret = crossfade_prepare(s, main_inlink, overlay_inlink, outlink, nb_samples, &main_buffer, &overlay_buffer, 0) < 0)
            return ret;
    } else if (s->transition_pts != AV_NOPTS_VALUE) {
        nb_samples = s->main_eof ? s->main_sample_buffers[0]->size : s->cf_samples;
        if (s->transition_pts < av_rescale_q(s->cf_samples, (AVRational){ 1, outlink->sample_rate }, outlink->time_base)) {
            nb_samples = av_rescale_q(s->transition_pts, outlink->time_base, (AVRational){ 1, outlink->sample_rate });
        }

        if (ret = crossfade_samples(s, main_inlink, overlay_inlink, outlink, nb_samples, &out, 0) < 0)
            return ret;

        return ff_filter_frame(outlink, out);
    } else {
        nb_samples = FFMIN(s->main_sample_buffers[0]->size, SEGMENT_SIZE);
        main_buffer = ff_get_audio_buffer(outlink, nb_samples);
        if (!main_buffer)
            return AVERROR(ENOMEM);

        main_buffer->pts = s->pts;
        s->pts += av_rescale_q(nb_samples, (AVRational){ 1, outlink->sample_rate }, outlink->time_base);

        if (ret = ring_remove(s->main_sample_buffers, main_inlink, main_buffer->extended_data, nb_samples) < 0)
            return ret;
    }

    if (!ff_inlink_evaluate_timeline_at_frame(main_inlink, main_buffer) || (s->overlay_eof && s->overlay_sample_buffers[0]->size == 0)) {
        return ff_filter_frame(outlink, main_buffer);
    } else {
        if (s->transition_pts == AV_NOPTS_VALUE) {
            nb_samples = FFMIN(s->overlay_sample_buffers[0]->size, SEGMENT_SIZE);
            overlay_buffer = ff_get_audio_buffer(outlink, nb_samples);
            if (!overlay_buffer)
                return AVERROR(ENOMEM);
            if (ret = ring_remove(s->overlay_sample_buffers, overlay_inlink, overlay_buffer->extended_data, nb_samples) < 0)
                return ret;

            overlay_buffer->pts = main_buffer->pts;
        }
        av_frame_free(&main_buffer);
        return ff_filter_frame(outlink, overlay_buffer);
    }
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AOverlayContext *s = ctx->priv;
    int ret, size, ring_buffer_size;

    switch (outlink->format) {
    case AV_SAMPLE_FMT_DBLP: s->crossfade_samples = crossfade_samples_dblp;
                             size = sizeof(double);
                             break;
    case AV_SAMPLE_FMT_FLTP: s->crossfade_samples = crossfade_samples_fltp;
                             size = sizeof(float);
                             break;
    case AV_SAMPLE_FMT_S16P: s->crossfade_samples = crossfade_samples_s16p;
                             size = sizeof(int16_t);
                             break;
    case AV_SAMPLE_FMT_S32P: s->crossfade_samples = crossfade_samples_s32p;
                             size = sizeof(int32_t);
                             break;
    }

    if (s->cf_duration)
        s->cf_samples = av_rescale(s->cf_duration, outlink->sample_rate, AV_TIME_BASE);
    else
        s->cf_samples = av_rescale(100000, outlink->sample_rate, AV_TIME_BASE);

    s->nb_channels = outlink->ch_layout.nb_channels;

    s->cf0 = av_malloc_array(s->nb_channels, sizeof(uint8_t*));
    s->cf1 = av_malloc_array(s->nb_channels, sizeof(uint8_t*));

    ring_buffer_size = SEGMENT_SIZE + SEGMENT_SIZE * (1 + ((s->cf_samples - 1) / SEGMENT_SIZE));

    s->main_sample_buffers = av_malloc_array(s->nb_channels, sizeof(FrameRingBuffer*));
    for (int i = 0; i < s->nb_channels; i++) {
        s->cf0[i] = av_malloc_array(s->cf_samples, size);
        ret = ring_init(&s->main_sample_buffers[i], ring_buffer_size, size);
        if (ret < 0)
            return ret;
    }

    s->overlay_sample_buffers = av_malloc_array(s->nb_channels, sizeof(FrameRingBuffer*));
    for (int i = 0; i < s->nb_channels; i++) {
        s->cf1[i] = av_malloc_array(s->cf_samples, size);
        ret = ring_init(&s->overlay_sample_buffers[i], ring_buffer_size, size);
        if (ret < 0)
            return ret;
    }

    return 0;
}

static const AVFilterPad avfilter_af_aoverlay_inputs[] = {
    {
        .name = "main",
        .type = AVMEDIA_TYPE_AUDIO,
    },
    {
        .name = "s->overlay_input",
        .type = AVMEDIA_TYPE_AUDIO,
    },
};

static const AVFilterPad avfilter_af_aoverlay_outputs[] = {
    {
        .name           = "default",
        .type           = AVMEDIA_TYPE_AUDIO,
        .config_props   = config_output,
    },
};

const AVFilter ff_af_aoverlay = {
    .name           = "aoverlay",
    .description    = NULL_IF_CONFIG_SMALL("Replace a specified section of an audio stream with another audio input."),
    .priv_size      = sizeof(AOverlayContext),
    .priv_class     = &aoverlay_class,
    .activate       = activate,
    .init           = init,
    .uninit         = uninit,
    FILTER_INPUTS(avfilter_af_aoverlay_inputs),
    FILTER_OUTPUTS(avfilter_af_aoverlay_outputs),
    FILTER_QUERY_FUNC(query_formats),
    .flags          = AVFILTER_FLAG_SUPPORT_TIMELINE_INTERNAL,
};