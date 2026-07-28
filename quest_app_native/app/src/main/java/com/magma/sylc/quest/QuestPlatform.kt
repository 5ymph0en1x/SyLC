package com.magma.sylc.quest

import android.content.Context
import android.os.Build

object QuestPlatform {
    private const val OCULUS_HAND_TRACKING = "oculus.software.handtracking"
    private const val OCULUS_BOUNDARYLESS = "com.oculus.feature.BOUNDARYLESS_APP"

    fun isQuest(context: Context): Boolean {
        val manufacturer = Build.MANUFACTURER.lowercase()
        val model = Build.MODEL.lowercase()
        val packageManager = context.packageManager
        return manufacturer.contains("oculus") ||
            manufacturer.contains("meta") ||
            model.contains("quest") ||
            packageManager.hasSystemFeature(OCULUS_HAND_TRACKING) ||
            packageManager.hasSystemFeature(OCULUS_BOUNDARYLESS)
    }
}
