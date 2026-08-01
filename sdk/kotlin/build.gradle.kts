plugins {
    kotlin("multiplatform") version "2.1.10"
    kotlin("plugin.serialization") version "2.1.10"
}

repositories {
    mavenCentral()
}

kotlin {
    val hostOs = System.getProperty("os.name")
    val isMingwX64 = hostOs.startsWith("Windows")
    val nativeTarget = when {
        hostOs == "Mac OS X" -> macosArm64("native") // Defaulting to arm64 for Apple Silicon
        hostOs == "Linux" -> linuxX64("native")
        isMingwX64 -> mingwX64("native")
        else -> throw GradleException("Host OS is not supported in Kotlin/Native.")
    }

    nativeTarget.apply {
        compilations.getByName("main") {
            cinterops {
                val libravdb by creating {
                    defFile(project.file("src/nativeInterop/cinterop/libravdb.def"))
                    headers(project.file("../cgo/libravdb.h"))
                    includeDirs(project.file("../cgo"))
                }
            }
        }
        binaries.all {
            linkerOpts("-L${project.file("../cgo").absolutePath}", "-lravdb", "-rpath", project.file("../cgo").absolutePath)
        }
    }
    
    tasks.withType<org.jetbrains.kotlin.gradle.targets.native.tasks.KotlinNativeTest> {
        environment("DYLD_LIBRARY_PATH", project.file("../cgo").absolutePath)
    }
    
    sourceSets {
        val nativeMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
            }
        }
        val nativeTest by getting
    }
}
