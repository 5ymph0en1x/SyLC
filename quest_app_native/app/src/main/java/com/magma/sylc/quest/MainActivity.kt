package com.magma.sylc.quest

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.Surface
import android.view.SurfaceHolder
import android.view.View
import android.view.WindowManager
import android.view.animation.AnimationUtils
import android.widget.FrameLayout
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.core.view.doOnLayout
import androidx.xr.runtime.Session
import androidx.xr.runtime.SessionCreateSuccess
import androidx.xr.scenecore.scene
import com.magma.sylc.quest.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private lateinit var binding: ActivityMainBinding
    private var session: CastSession? = null
    private var videoSurface: Surface? = null
    private var connectionPhase = ConnectionPhase.DISCONNECTED
    private var videoTimeoutJob: Job? = null
    private var isFullscreen = false
    private var questStereoLaunchPending = false
    private var androidXrSession: Session? = null
    @Volatile private var hasReceivedVideo = false

    private val scope = CoroutineScope(Dispatchers.Main + Job())

    enum class ViewMode { MONO, TWO_D, THREE_D }
    private enum class ConnectionPhase { DISCONNECTED, CONNECTING, CONNECTED, STREAMING }

    private var currentMode = ViewMode.THREE_D
        set(value) {
            field = value
            applyViewMode(value)
        }

    private var lastUpdateNano = 0L
    private var frameCount = 0
    private var byteCount = 0L

    // Where the PC lives. Defaults to 127.0.0.1, which is the USB-C path via
    // `adb reverse tcp:47420 tcp:47420`, but is user-editable and persisted so
    // the receiver also works over Wi-Fi (the sender binds 0.0.0.0).
    private val settings by lazy { CastSettings(this) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        initializeAndroidXrSession()
        // AndroidX's dispatcher, not android.window.OnBackInvokedDispatcher:
        // the platform one is API 33, which would crash on every phone running
        // Android 8-12. This one covers the whole minSdk range and still wires
        // up predictive back on 33+ by itself.
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (isFullscreen) {
                    setFullscreen(false)
                } else {
                    finishAfterTransition()
                }
            }
        })

        binding.videoSurface.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                val announced = settings.lastStreamFormat
                holder.setFixedSize(announced.width, announced.height)
                videoSurface = holder.surface
            }

            override fun surfaceChanged(
                holder: SurfaceHolder,
                format: Int,
                width: Int,
                height: Int
            ) = Unit

            override fun surfaceDestroyed(holder: SurfaceHolder) {
                videoSurface = null
                disconnect()
            }
        })

        bindSenderAddressControls()

        binding.playButton.setOnClickListener {
            Log.w(TAG, "USER EVENT: Play clicked")
            if (connectionPhase == ConnectionPhase.DISCONNECTED) {
                if (currentMode == ViewMode.THREE_D && QuestPlatform.isQuest(this)) {
                    launchQuestStereo()
                } else {
                    connect()
                }
            } else {
                Toast.makeText(this, R.string.already_connected, Toast.LENGTH_SHORT).show()
            }
            animateButton(it)
        }

        binding.stopButton.setOnClickListener {
            Log.w(TAG, "USER EVENT: Stop clicked")
            if (connectionPhase != ConnectionPhase.DISCONNECTED) disconnect()
            animateButton(it)
        }

        binding.fullscreenButton.setOnClickListener {
            Log.w(TAG, "USER EVENT: Fullscreen enabled")
            if (currentMode == ViewMode.THREE_D && QuestPlatform.isQuest(this)) {
                launchQuestStereo()
            } else {
                setFullscreen(true)
            }
            it.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
            animateButton(it)
        }

        binding.exitFullscreenButton.setOnClickListener {
            Log.w(TAG, "USER EVENT: Fullscreen disabled")
            setFullscreen(false)
            it.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
            animateButton(it)
        }

        binding.videoContainer.setOnClickListener {
            if (isFullscreen) {
                Log.w(TAG, "USER EVENT: Fullscreen disabled from video surface")
                setFullscreen(false)
                it.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
            }
        }

        // A real single-selection control keeps every mode focusable, checkable and
        // touchable on both classic Android windows and Android XR panels.
        binding.modeToggleGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener

            val mode = when (checkedId) {
                R.id.btnMono -> ViewMode.MONO
                R.id.btn2D -> ViewMode.TWO_D
                R.id.btn3D -> ViewMode.THREE_D
                else -> return@addOnButtonCheckedListener
            }
            Log.w(TAG, "USER EVENT: ${mode.name} mode selected")
            currentMode = mode

            binding.modeToggleGroup.findViewById<View>(checkedId)?.let { button ->
                button.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
                animateButton(button)
            }
            if (
                mode == ViewMode.THREE_D &&
                QuestPlatform.isQuest(this) &&
                connectionPhase != ConnectionPhase.DISCONNECTED
            ) {
                launchQuestStereo()
            }
        }

        val restoredMode = savedInstanceState
            ?.getString(STATE_VIEW_MODE)
            ?.let { saved -> ViewMode.entries.firstOrNull { it.name == saved } }
            ?: defaultViewMode()
        binding.modeToggleGroup.check(restoredMode.buttonId)
        currentMode = restoredMode
        binding.stopButton.isEnabled = false
        setFullscreen(savedInstanceState?.getBoolean(STATE_FULLSCREEN) == true, announce = false)

        binding.controlPanel.alpha = 0f
        binding.controlPanel.translationY = 12f
        binding.controlPanel.animate()
            .alpha(1f)
            .translationY(0f)
            .setDuration(280)
            .start()
    }

    private fun animateButton(button: View) {
        button.startAnimation(AnimationUtils.loadAnimation(this, R.anim.scale_bounce))
    }

    /**
     * A headset shows the two eyes separately; a phone screen cannot. Opening a
     * phone in 3D would present the raw side-by-side frame — two squashed
     * half-images — which reads as a broken stream rather than a stereo one.
     * Mono crops to the left eye, so the phone simply shows the film. The three
     * modes stay available either way; only the landing mode differs.
     */
    private fun defaultViewMode(): ViewMode =
        if (QuestPlatform.isQuest(this)) ViewMode.THREE_D else ViewMode.MONO

    private fun launchQuestStereo() {
        if (questStereoLaunchPending) return
        questStereoLaunchPending = true
        Log.i(TAG, "Launching Meta Quest stereoscopic receiver")
        if (connectionPhase != ConnectionPhase.DISCONNECTED) {
            disconnect()
        }
        startActivity(
            Intent(this, QuestStereoActivity::class.java)
                .setAction(Intent.ACTION_MAIN)
        )
    }

    private fun initializeAndroidXrSession() {
        if (!packageManager.hasSystemFeature(ANDROID_XR_SPATIAL_FEATURE)) return

        scope.launch {
            try {
                when (
                    val result = Session.create(
                        context = this@MainActivity,
                        lifecycleOwner = this@MainActivity
                    )
                ) {
                    is SessionCreateSuccess -> {
                        androidXrSession = result.session
                        Log.i(TAG, "Android XR spatial session ready")
                        requestAndroidXrSpace(isFullscreen)
                    }

                    else -> Log.w(
                        TAG,
                        "Android XR spatial session unavailable: ${result::class.simpleName}"
                    )
                }
            } catch (error: RuntimeException) {
                Log.w(TAG, "Android XR spatial session initialization failed", error)
            }
        }
    }

    private fun setFullscreen(enabled: Boolean, announce: Boolean = true) {
        isFullscreen = enabled
        binding.controlPanel.visibility = if (enabled) View.GONE else View.VISIBLE
        // Keep the overlay measured while hidden. Android XR can otherwise draw a
        // newly-visible constrained view but leave its spatial hit target at 0×0.
        binding.exitFullscreenButton.visibility =
            if (enabled) View.VISIBLE else View.INVISIBLE
        binding.videoContainer.contentDescription =
            if (enabled) getString(R.string.action_exit_fullscreen_description) else null
        binding.videoContainer.isClickable = enabled
        binding.videoContainer.isFocusable = enabled

        WindowCompat.getInsetsController(window, binding.root).apply {
            systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            if (enabled) {
                hide(WindowInsetsCompat.Type.systemBars())
            } else {
                show(WindowInsetsCompat.Type.systemBars())
            }
        }
        requestAndroidXrSpace(enabled)

        binding.videoContainer.requestLayout()
        binding.exitFullscreenButton.bringToFront()
        if (announce) {
            ViewCompat.setAccessibilityPaneTitle(
                binding.root,
                getString(
                    if (enabled) {
                        R.string.fullscreen_enabled_announcement
                    } else {
                        R.string.fullscreen_disabled_announcement
                    }
                )
            )
        }
    }

    private fun requestAndroidXrSpace(fullSpace: Boolean) {
        androidXrSession?.scene?.let { scene ->
            Log.i(TAG, "Requesting Android XR ${if (fullSpace) "Full" else "Home"} Space")
            if (fullSpace) {
                scene.requestFullSpace()
            } else {
                scene.requestHomeSpace()
            }
        }
    }

    private fun applyViewMode(mode: ViewMode) {
        val container = binding.videoContainer
        val containerParams = container.layoutParams as ConstraintLayout.LayoutParams

        when (mode) {
            ViewMode.MONO -> {
                containerParams.dimensionRatio = "16:9"
                binding.modeBadge.setText(R.string.mode_badge_mono)
                binding.modeDescriptionText.setText(R.string.mode_mono_description)
            }

            ViewMode.TWO_D -> {
                containerParams.dimensionRatio = "32:9"
                binding.modeBadge.setText(R.string.mode_badge_2d)
                binding.modeDescriptionText.setText(R.string.mode_2d_description)
            }

            ViewMode.THREE_D -> {
                containerParams.dimensionRatio = "32:9"
                binding.modeBadge.setText(R.string.mode_badge_3d)
                binding.modeDescriptionText.setText(R.string.mode_3d_description)
            }
        }

        container.layoutParams = containerParams
        container.requestLayout()

        // Wait for the ratio change to be measured. Mono doubles the SBS surface
        // and lets the FrameLayout clip it to the left eye; 2D/3D preserve SBS.
        container.doOnLayout { laidOutContainer ->
            val surfaceParams =
                binding.videoSurface.layoutParams as FrameLayout.LayoutParams
            surfaceParams.width = if (mode == ViewMode.MONO) {
                laidOutContainer.width * 2
            } else {
                FrameLayout.LayoutParams.MATCH_PARENT
            }
            surfaceParams.height = FrameLayout.LayoutParams.MATCH_PARENT
            surfaceParams.gravity = Gravity.START or Gravity.CENTER_VERTICAL
            binding.videoSurface.layoutParams = surfaceParams
            binding.videoSurface.translationX = 0f
        }

        binding.videoSurface.contentDescription = getString(mode.descriptionId)
        binding.controlPanel.bringToFront()
    }

    /**
     * The PC address is configuration, not a constant. It is shown, editable and
     * persisted; "Find PC" broadcasts a HELLO and fills in whoever answers, so
     * the address rarely has to be typed at all.
     */
    private fun bindSenderAddressControls() {
        binding.senderHostInput.setText(settings.host)
        binding.senderHostInput.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) commitSenderAddress()
        }
        binding.senderHostInput.setOnEditorActionListener { _, _, _ ->
            commitSenderAddress()
            false
        }
        binding.discoverButton.setOnClickListener { view ->
            animateButton(view)
            discoverSender()
        }
    }

    private fun commitSenderAddress() {
        val typed = binding.senderHostInput.text?.toString().orEmpty()
        settings.host = typed
        // Echo back what was actually stored, so a blank entry visibly reverts
        // to the USB-C default instead of leaving a misleading empty field.
        if (binding.senderHostInput.text?.toString() != settings.host) {
            binding.senderHostInput.setText(settings.host)
        }
    }

    private fun discoverSender() {
        // Discovery is a Wi-Fi affair. Over USB-C the sender is reached through
        // `adb reverse` on the headset's own loopback, and it runs a TCP server
        // with no UDP listener at all -- so a broadcast can never be answered.
        // Say that plainly instead of letting the search fail as if something
        // were wrong.
        if (settings.host.isLoopbackAddress()) {
            binding.connectionHintText.setText(R.string.discovery_usb)
            return
        }
        binding.discoverButton.isEnabled = false
        binding.connectionHintText.setText(R.string.discovery_searching)
        scope.launch {
            val found = CastDiscovery.discover(settings.port)
            binding.discoverButton.isEnabled = true
            if (found == null) {
                binding.connectionHintText.setText(R.string.discovery_none)
                return@launch
            }
            settings.host = found.host
            settings.port = found.port
            binding.senderHostInput.setText(found.host)
            binding.connectionHintText.text =
                getString(R.string.discovery_found, found.host)
        }
    }

    private fun String.isLoopbackAddress(): Boolean =
        this == "localhost" || startsWith("127.")

    private fun connect() {
        val surface = videoSurface
        if (surface == null || !surface.isValid) {
            Toast.makeText(this, R.string.decoder_not_ready, Toast.LENGTH_SHORT).show()
            return
        }

        binding.statusText.setText(R.string.status_connecting)
        binding.connectionHintText.setText(R.string.status_hint_connecting)
        binding.statusDot.setBackgroundResource(R.drawable.status_dot_off)
        binding.playButton.isEnabled = false
        binding.stopButton.isEnabled = true
        connectionPhase = ConnectionPhase.CONNECTING
        hasReceivedVideo = false

        lastUpdateNano = System.nanoTime()
        frameCount = 0
        byteCount = 0

        session = CastSession(settings, SessionListener()).also {
            it.start(surface, scope)
        }
    }

    /**
     * CastSession owns the transport, decoder and audio; this only turns its
     * events into what the panel shows. Callbacks arrive off the main thread.
     */
    private inner class SessionListener : CastSession.Listener {

        override fun onFirstVideoUnit(lengthBytes: Int) {
            hasReceivedVideo = true
            runOnUiThread {
                if (connectionPhase != ConnectionPhase.DISCONNECTED) {
                    binding.statusText.setText(R.string.status_decoding)
                    binding.connectionHintText.setText(R.string.status_hint_decoding)
                }
            }
        }

        override fun onVideoUnit(lengthBytes: Int) {
            frameCount++
            byteCount += lengthBytes
            val now = System.nanoTime()
            if (now - lastUpdateNano < STATS_INTERVAL_NANOS) return
            val dt = (now - lastUpdateNano) / NANOS_PER_SECOND
            val fps = frameCount / dt
            val mbps = (byteCount * 8) / (dt * BITS_PER_MEGABIT)
            frameCount = 0
            byteCount = 0
            lastUpdateNano = now
            runOnUiThread {
                binding.statsText.text = getString(R.string.stats_format, mbps, fps)
            }
        }

        override fun onFirstFrameRendered() {
            runOnUiThread {
                if (connectionPhase != ConnectionPhase.DISCONNECTED) {
                    videoTimeoutJob?.cancel()
                    connectionPhase = ConnectionPhase.STREAMING
                    binding.statusText.setText(R.string.status_connected)
                    binding.connectionHintText.setText(R.string.status_hint_connected)
                    binding.statusDot.setBackgroundResource(R.drawable.status_dot)
                }
            }
        }

        override fun onConnected() {
            runOnUiThread {
                if (connectionPhase == ConnectionPhase.DISCONNECTED) return@runOnUiThread
                val resuming = connectionPhase == ConnectionPhase.STREAMING
                connectionPhase = if (resuming) {
                    ConnectionPhase.STREAMING
                } else {
                    ConnectionPhase.CONNECTED
                }
                binding.statusText.setText(
                    if (resuming) R.string.status_connected
                    else R.string.status_waiting_video
                )
                binding.connectionHintText.setText(
                    if (resuming) R.string.status_hint_connected
                    else R.string.status_hint_waiting_video
                )
                binding.statusDot.setBackgroundResource(R.drawable.status_dot)
                if (!resuming) startVideoTimeout()
            }
        }

        override fun onConnectionError(error: Throwable) {
            Log.w(TAG, "Cast connection interrupted; automatic retry active", error)
            runOnUiThread {
                if (connectionPhase != ConnectionPhase.DISCONNECTED) {
                    binding.statusText.setText(R.string.status_connection_lost)
                    binding.connectionHintText.setText(R.string.status_hint_connection_lost)
                    binding.statusDot.setBackgroundResource(R.drawable.status_dot_warning)
                }
            }
        }

        override fun onDecoderError(message: String) {
            runOnUiThread {
                if (connectionPhase != ConnectionPhase.DISCONNECTED) {
                    videoTimeoutJob?.cancel()
                    binding.statusText.setText(R.string.status_decoder_error)
                    binding.connectionHintText.text = message
                    binding.statusDot.setBackgroundResource(R.drawable.status_dot_warning)
                }
            }
        }

        override fun onStopped(error: Throwable?) {
            runOnUiThread { onDisconnected(error) }
        }
    }

    private fun startVideoTimeout() {
        videoTimeoutJob?.cancel()
        videoTimeoutJob = scope.launch {
            delay(VIDEO_START_TIMEOUT_MS)
            if (connectionPhase == ConnectionPhase.STREAMING ||
                connectionPhase == ConnectionPhase.DISCONNECTED
            ) {
                return@launch
            }

            binding.statusDot.setBackgroundResource(R.drawable.status_dot_warning)
            if (hasReceivedVideo) {
                binding.statusText.setText(R.string.status_decoder_error)
                binding.connectionHintText.setText(R.string.status_hint_decode_timeout)
            } else {
                binding.statusText.setText(R.string.status_no_video)
                binding.connectionHintText.setText(R.string.status_hint_no_video)
            }
        }
    }

    private fun disconnect() {
        videoTimeoutJob?.cancel()
        session?.stop()
    }

    private fun onDisconnected(error: Throwable? = null) {
        videoTimeoutJob?.cancel()
        connectionPhase = ConnectionPhase.DISCONNECTED
        hasReceivedVideo = false
        if (error == null) {
            binding.statusText.setText(R.string.status_disconnected)
            binding.connectionHintText.setText(R.string.status_hint_ready)
            binding.statusDot.setBackgroundResource(R.drawable.status_dot_off)
        } else {
            binding.statusText.setText(R.string.status_connection_lost)
            binding.connectionHintText.setText(R.string.status_hint_connection_lost)
            binding.statusDot.setBackgroundResource(R.drawable.status_dot_warning)
        }
        binding.statsText.setText(R.string.stats_empty)
        binding.playButton.isEnabled = true
        binding.stopButton.isEnabled = false
        // Releases MediaCodec and AudioTrack; idempotent, so the stop button and
        // the transport's own end-of-loop can both land here safely.
        session?.stop()
        session = null
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putString(STATE_VIEW_MODE, currentMode.name)
        outState.putBoolean(STATE_FULLSCREEN, isFullscreen)
        super.onSaveInstanceState(outState)
    }

    override fun onDestroy() {
        disconnect()
        scope.cancel()
        super.onDestroy()
    }

    override fun onResume() {
        super.onResume()
        questStereoLaunchPending = false
    }

    private val ViewMode.buttonId: Int
        get() = when (this) {
            ViewMode.MONO -> R.id.btnMono
            ViewMode.TWO_D -> R.id.btn2D
            ViewMode.THREE_D -> R.id.btn3D
        }

    private val ViewMode.displayName: String
        get() = when (this) {
            ViewMode.MONO -> getString(R.string.mode_mono)
            ViewMode.TWO_D -> getString(R.string.mode_2d)
            ViewMode.THREE_D -> getString(R.string.mode_3d)
        }

    private val ViewMode.descriptionId: Int
        get() = when (this) {
            ViewMode.MONO -> R.string.mode_mono_description
            ViewMode.TWO_D -> R.string.mode_2d_description
            ViewMode.THREE_D -> R.string.mode_3d_description
        }

    companion object {
        private const val TAG = "MainActivity"
        private const val STATE_VIEW_MODE = "view_mode"
        private const val STATE_FULLSCREEN = "fullscreen"
        private const val VIDEO_START_TIMEOUT_MS = 3_000L
        private const val STATS_INTERVAL_NANOS = 1_000_000_000L
        private const val NANOS_PER_SECOND = 1_000_000_000.0
        private const val BITS_PER_MEGABIT = 1_000_000.0
        private const val ANDROID_XR_SPATIAL_FEATURE = "android.software.xr.api.spatial"
    }
}
