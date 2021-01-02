package com.kelompoksatu.sentimenanalisis

import android.app.Application
import com.kelompoksatu.sentimenanalisis.di.viewModelModule
import org.koin.android.ext.koin.androidContext
import org.koin.core.context.startKoin

class AppController : Application() {

    override fun onCreate() {
        super.onCreate()
        startKoin {
            androidContext(this@AppController)
            modules(viewModelModule)
        }

    }

}