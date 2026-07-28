package com.magma.sylc.quest

import android.content.Context
import android.content.SharedPreferences

/**
 * Persisted receiver configuration.
 *
 * The PC address used to be a compile-time `127.0.0.1`, which only resolves to
 * the sender when the headset is tethered and `adb reverse tcp:47420 tcp:47420`
 * is active. The sender binds `0.0.0.0`, so it is reachable over Wi-Fi too --
 * it just needs to be told where it lives. That default is kept (USB-C remains
 * the best link) but it is now a starting value, not a wall.
 */
class CastSettings(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    var host: String
        get() = prefs.getString(KEY_HOST, DEFAULT_HOST)?.takeIf { it.isNotBlank() }
            ?: DEFAULT_HOST
        set(value) {
            val cleaned = value.trim()
            prefs.edit().putString(
                KEY_HOST,
                if (cleaned.isEmpty()) DEFAULT_HOST else cleaned
            ).apply()
        }

    var port: Int
        get() = prefs.getInt(KEY_PORT, DEFAULT_PORT).takeIf { it in 1..65535 } ?: DEFAULT_PORT
        set(value) {
            if (value in 1..65535) prefs.edit().putInt(KEY_PORT, value).apply()
        }

    /** Screen width in metres, as seen in the headset. */
    var panelWidthMeters: Float
        get() = prefs.getFloat(KEY_PANEL_WIDTH, DEFAULT_PANEL_WIDTH)
            .coerceIn(MIN_PANEL_WIDTH, MAX_PANEL_WIDTH)
        set(value) {
            prefs.edit().putFloat(
                KEY_PANEL_WIDTH, value.coerceIn(MIN_PANEL_WIDTH, MAX_PANEL_WIDTH)
            ).apply()
        }

    /** Distance from the viewer to the screen, in metres. */
    var panelDistanceMeters: Float
        get() = prefs.getFloat(KEY_PANEL_DISTANCE, DEFAULT_PANEL_DISTANCE)
            .coerceIn(MIN_PANEL_DISTANCE, MAX_PANEL_DISTANCE)
        set(value) {
            prefs.edit().putFloat(
                KEY_PANEL_DISTANCE, value.coerceIn(MIN_PANEL_DISTANCE, MAX_PANEL_DISTANCE)
            ).apply()
        }

    /** Passthrough on shows the room; off gives a dark room, which suits film. */
    var passthroughEnabled: Boolean
        get() = prefs.getBoolean(KEY_PASSTHROUGH, DEFAULT_PASSTHROUGH)
        set(value) = prefs.edit().putBoolean(KEY_PASSTHROUGH, value).apply()

    /**
     * The stream description last announced by the sender.
     *
     * The decoder adapts to a new announcement immediately, but a panel's pixel
     * size is fixed when the panel is built -- before any handshake has
     * happened. Remembering the last announcement lets the screen be created at
     * the right resolution from the second session onwards; a mismatch in the
     * meantime costs sharpness (the surface scales), never correctness, since
     * the left/right split is proportional.
     */
    var lastStreamFormat: SyLcProtocol.StreamFormat
        get() = SyLcProtocol.StreamFormat(
            width = prefs.getInt(KEY_STREAM_WIDTH, SyLcProtocol.StreamFormat.DEFAULT.width),
            height = prefs.getInt(KEY_STREAM_HEIGHT, SyLcProtocol.StreamFormat.DEFAULT.height),
            fps = prefs.getInt(KEY_STREAM_FPS, SyLcProtocol.StreamFormat.DEFAULT.fps),
            stereo = prefs.getString(KEY_STREAM_STEREO, SyLcProtocol.StreamFormat.DEFAULT.stereo)
                ?: SyLcProtocol.StreamFormat.DEFAULT.stereo,
        )
        set(value) {
            prefs.edit()
                .putInt(KEY_STREAM_WIDTH, value.width)
                .putInt(KEY_STREAM_HEIGHT, value.height)
                .putInt(KEY_STREAM_FPS, value.fps)
                .putString(KEY_STREAM_STEREO, value.stereo)
                .apply()
        }

    /** Horizontal field of view the current screen geometry subtends, degrees. */
    fun horizontalFovDegrees(): Float {
        val halfAngle = Math.atan2(
            (panelWidthMeters / 2.0), panelDistanceMeters.toDouble()
        )
        return Math.toDegrees(2.0 * halfAngle).toFloat()
    }

    companion object {
        private const val PREFS_NAME = "sylc_cast"
        private const val KEY_HOST = "pc_host"
        private const val KEY_PORT = "pc_port"
        private const val KEY_PANEL_WIDTH = "panel_width_m"
        private const val KEY_PANEL_DISTANCE = "panel_distance_m"
        private const val KEY_PASSTHROUGH = "passthrough"
        private const val KEY_STREAM_WIDTH = "stream_width"
        private const val KEY_STREAM_HEIGHT = "stream_height"
        private const val KEY_STREAM_FPS = "stream_fps"
        private const val KEY_STREAM_STEREO = "stream_stereo"

        /** Matches `adb reverse tcp:47420 tcp:47420` over USB-C. */
        const val DEFAULT_HOST = "127.0.0.1"
        const val DEFAULT_PORT = 47420

        // A 3.2 m screen at 3.0 m is about 56 deg wide -- a cinema seat rather
        // than the 1.6 m / 2.0 m television the receiver used to pin us to.
        const val DEFAULT_PANEL_WIDTH = 3.2f
        const val DEFAULT_PANEL_DISTANCE = 3.0f
        const val MIN_PANEL_WIDTH = 1.0f
        const val MAX_PANEL_WIDTH = 12.0f
        const val MIN_PANEL_DISTANCE = 1.0f
        const val MAX_PANEL_DISTANCE = 12.0f
        const val DEFAULT_PASSTHROUGH = false
    }
}
