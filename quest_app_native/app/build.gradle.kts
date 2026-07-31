plugins {
    id("com.android.application")
}

android {
    namespace = "com.magma.sylc.quest"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "com.magma.sylc.quest"
        // 26 (Android 8.0), not 33: one APK for the Quest AND for phones.
        // minSdk is the ONE thing that actually blocks a phone install — the
        // package installer ignores <uses-feature>, and every Quest-only feature
        // here is already declared required="false". 26 is the true floor and it
        // is a resource constraint, not a dependency one: <adaptive-icon> needs
        // 26, while the Meta Spatial and androidx.xr AARs merge happily as low
        // as 24. Quest 3 runs Android 12+, so nothing about it changes.
        minSdk = 26
        // 34, not 36: Horizon OS accepts 32-36 for a 2D panel app but only
        // 32-34 for an immersive one, and QuestStereoActivity makes this an
        // immersive app. The store rejects the upload above 34. Nothing here
        // needs an API above 34; compileSdk stays at 36 so the newer AndroidX
        // artifacts still compile.
        targetSdk = 34
        versionCode = 5
        versionName = "1.3"
    }

    signingConfigs {
        // Persistent self-signed release key (quest_app_native/sylc-release.keystore).
        // NOT the SDK debug key: Play Protect on Android 14+ hard-blocks
        // never-seen apps signed with the well-known debug certificate when
        // they arrive via messaging apps ("Appli bloquée pour protéger votre
        // appareil", and the install session dies even after "Installer quand
        // même"). A stable release identity + non-debuggable build restores
        // the normal sideload flow. The password is deliberately plain — this
        // key asserts UPDATE CONTINUITY across our own devices, not secrecy.
        create("release") {
            storeFile = rootProject.file("sylc-release.keystore")
            storePassword = "sylc-cast-release"
            keyAlias = "sylc"
            keyPassword = "sylc-cast-release"
        }
    }

    buildTypes {
        release {
            // Horizon OS is arm64-only, so the armeabi-v7a / x86 / x86_64
            // slices the Meta Spatial, ARCore and androidx.xr AARs contribute
            // are pure download weight on every device that can run this app
            // (the store flags the 32-bit ones outright). Filtering on the
            // release build only keeps x86_64 available in debug, which is what
            // the Android XR emulator needs to load those same libraries.
            ndk {
                abiFilters += "arm64-v8a"
            }
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            // The release keystore is NOT distributed (it asserts update
            // continuity across the author's devices). Public checkouts fall
            // back to the debug key so `assembleRelease` still builds.
            signingConfig = if (rootProject.file("sylc-release.keystore").exists())
                signingConfigs.getByName("release")
            else
                signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        viewBinding = true
    }
    testOptions {
        unitTests {
            // Protocol parsing logs through android.util.Log on its fallback
            // paths; without this the stub throws and hides the real assertion.
            isReturnDefaultValues = true
        }
    }
    packaging {
        jniLibs {
            keepDebugSymbols += setOf(
                "**/libAetherGlobals.so",
                "**/libMetaSpatialSDK*.so",
                "**/libopenxr_loader.so",
                "**/libthird-party_zlib_1_3_1_z.so",
                "**/libandroidx.xr.*.so",
                "**/libarcore_sdk_*.so",
                "**/libimpress_api_jni.so"
            )
        }
    }
}

dependencies {
    // Unit tests run on the JVM, where android.jar's org.json is a stub that
    // throws. The real implementation makes the protocol tests meaningful.
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")

    implementation("androidx.core:core-ktx:1.18.0")
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("com.google.android.material:material:1.14.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.11.0")

    // Android XR emulator: Home Space / Full Space transitions.
    implementation("androidx.xr.runtime:runtime:1.0.0-beta01")
    implementation("androidx.xr.scenecore:scenecore:1.0.0-beta01")

    // Native immersive rendering for Meta Quest / Horizon OS.
    implementation("com.meta.spatial:meta-spatial-sdk:0.13.2")
    implementation("com.meta.spatial:meta-spatial-sdk-toolkit:0.13.2")
    implementation("com.meta.spatial:meta-spatial-sdk-vr:0.13.2")
}
