package com.magma.sylc.quest

import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.TextView
import com.google.android.material.button.MaterialButton
import com.meta.spatial.core.Entity
import com.meta.spatial.core.Pose
import com.meta.spatial.core.SpatialFeature
import com.meta.spatial.core.Vector3
import com.meta.spatial.runtime.PanelSceneObject
import com.meta.spatial.runtime.ReferenceSpace
import com.meta.spatial.runtime.SceneObject
import com.meta.spatial.runtime.StereoMode
import com.meta.spatial.toolkit.AppSystemActivity
import com.meta.spatial.toolkit.DpDisplayOptions
import com.meta.spatial.toolkit.LayoutXMLPanelRegistration
import com.meta.spatial.toolkit.MediaPanelRenderOptions
import com.meta.spatial.toolkit.MediaPanelSettings
import com.meta.spatial.toolkit.Panel
import com.meta.spatial.toolkit.PanelRegistration
import com.meta.spatial.toolkit.PanelStyleOptions
import com.meta.spatial.toolkit.PixelDisplayOptions
import com.meta.spatial.toolkit.QuadShapeOptions
import com.meta.spatial.toolkit.SceneObjectSystem
import com.meta.spatial.toolkit.Transform
import com.meta.spatial.vr.VRFeature
import java.util.concurrent.CompletableFuture
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Quest 3 immersive receiver.
 *
 * Meta's StereoMode.LeftRight routes the left half of the 3840x1080 decoded
 * surface to the left eye and the right half to the right eye.
 */
class QuestStereoActivity : AppSystemActivity() {

    private val mainScope = CoroutineScope(Dispatchers.Main + Job())
    private var session: CastSession? = null
    private var videoPanel: PanelSceneObject? = null
    private var initialized = false
    private var statusText: TextView? = null
    private var statusDot: View? = null
    private var pendingStatus = R.string.quest_stereo_connecting
    private var pendingWarning = false
    private val settings by lazy { CastSettings(this) }

    override fun registerFeatures(): List<SpatialFeature> = listOf(VRFeature(this))

    override fun registerPanels(): List<PanelRegistration> {
        return listOf(
            LayoutXMLPanelRegistration(
                R.id.questStereoControlsPanel,
                layoutIdCreator = { R.layout.panel_quest_stereo_controls },
                settingsCreator = {
                    com.meta.spatial.toolkit.UIPanelSettings(
                        shape = QuadShapeOptions(width = 1.05f, height = 0.22f),
                        display = DpDisplayOptions(width = 460f, height = 96f, dpi = 600),
                        style = PanelStyleOptions(
                            themeResourceId = R.style.Theme_SyLC_SpatialPanel
                        ),
                    )
                },
                panelSetupWithRootView = { rootView, _, _ ->
                    statusText = rootView.findViewById(R.id.questStereoStatus)
                    statusDot = rootView.findViewById(R.id.questStereoStatusDot)
                    rootView.findViewById<MaterialButton>(R.id.questStereoCloseButton)
                        .setOnClickListener { finish() }
                    updateStatus(pendingStatus, pendingWarning)
                },
            )
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.i(TAG, "Starting Quest 3 stereoscopic receiver")
    }

    override fun onSceneReady() {
        super.onSceneReady()
        scene.setReferenceSpace(ReferenceSpace.LOCAL_FLOOR)
        scene.setViewOrigin(0f, 0f, 0f, 0f)
        // A film is watched in a dark room by default; passthrough is opt-in
        // rather than forced on, and survives across sessions.
        scene.enablePassthrough(settings.passthroughEnabled)
    }

    override fun onVRReady() {
        super.onVRReady()
        if (initialized) return
        initialized = true

        // Screen geometry is user configuration, not a constant: the previous
        // fixed 1.6 m at 2.0 m subtends only ~44 deg, a television in the room.
        val announced = settings.lastStreamFormat
        val screenWidth = settings.panelWidthMeters
        val screenDistance = settings.panelDistanceMeters
        val aspect = if (announced.isSideBySide) {
            // Both eyes share the frame, so one eye is half as wide.
            (announced.width / 2f) / announced.height.toFloat()
        } else {
            announced.width / announced.height.toFloat()
        }
        val screenHeight = screenWidth / aspect
        Log.i(
            TAG,
            "Screen ${"%.2f".format(screenWidth)}x${"%.2f".format(screenHeight)} m " +
                "at ${"%.2f".format(screenDistance)} m " +
                "(${"%.0f".format(settings.horizontalFovDegrees())} deg), " +
                "source ${announced.width}x${announced.height}@${announced.fps} " +
                "${announced.stereo}, passthrough=${settings.passthroughEnabled}"
        )

        val videoEntity = Entity.create(
            listOf(Transform(Pose(Vector3(0f, EYE_HEIGHT_METERS, screenDistance))))
        )
        val mediaSettings = MediaPanelSettings(
            shape = QuadShapeOptions(width = screenWidth, height = screenHeight),
            display = PixelDisplayOptions(width = announced.width, height = announced.height),
            rendering = MediaPanelRenderOptions(
                stereoMode = if (announced.isSideBySide) {
                    StereoMode.LeftRight
                } else {
                    StereoMode.None
                }
            ),
        )
        videoPanel = PanelSceneObject(
            scene,
            videoEntity,
            mediaSettings.toPanelConfigOptions(),
        ).also { panel ->
            systemManager.findSystem<SceneObjectSystem>().addSceneObject(
                videoEntity,
                CompletableFuture<SceneObject>().apply { complete(panel) },
            )
        }

        // Keep the controls just below the screen whatever size it is.
        Entity.create(
            listOf(
                Panel(R.id.questStereoControlsPanel),
                Transform(
                    Pose(
                        Vector3(
                            0f,
                            EYE_HEIGHT_METERS - screenHeight / 2f - 0.18f,
                            screenDistance - 0.2f,
                        )
                    )
                ),
            )
        )

        val outputSurface = videoPanel?.surface
        if (outputSurface == null) {
            updateStatus(R.string.quest_stereo_error, warning = true)
            return
        }

        session = CastSession(settings, SessionListener()).also {
            it.start(outputSurface, mainScope)
        }
        updateStatus(R.string.quest_stereo_connecting)
    }

    /**
     * Everything mechanical lives in CastSession; this only maps events to what
     * the viewer sees. Callbacks arrive off the main thread.
     */
    private inner class SessionListener : CastSession.Listener {
        override fun onConnected() {
            runOnUiThread { updateStatus(R.string.quest_stereo_waiting) }
        }

        override fun onFirstFrameRendered() {
            runOnUiThread {
                updateStatus(R.string.quest_stereo_active)
                Log.i(TAG, "First stereoscopic frame rendered: left/right eye routing active")
            }
        }

        override fun onDecoderError(message: String) {
            Log.e(TAG, "Stereo decoder error: $message")
            runOnUiThread { updateStatus(R.string.quest_stereo_error, warning = true) }
        }

        override fun onConnectionError(error: Throwable) {
            Log.e(TAG, "Quest stereo connection failed", error)
            runOnUiThread { updateStatus(R.string.quest_stereo_error, warning = true) }
        }

        override fun onStopped(error: Throwable?) {
            if (error != null) {
                runOnUiThread { updateStatus(R.string.quest_stereo_error, warning = true) }
            }
        }
    }

    private fun updateStatus(stringId: Int, warning: Boolean = false) {
        pendingStatus = stringId
        pendingWarning = warning
        statusText?.setText(stringId)
        statusDot?.setBackgroundResource(
            if (warning) R.drawable.status_dot_warning else R.drawable.status_dot
        )
    }

    override fun onDestroy() {
        session?.stop()
        session = null
        mainScope.cancel()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "QuestStereoActivity"
        /** Seated eye height; the screen centre is placed here. */
        private const val EYE_HEIGHT_METERS = 1.45f
    }
}
