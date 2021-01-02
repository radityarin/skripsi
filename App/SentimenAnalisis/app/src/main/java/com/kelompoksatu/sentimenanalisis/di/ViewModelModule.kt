package com.kelompoksatu.sentimenanalisis.di
import com.kelompoksatu.sentimenanalisis.ui.diagnosis.viewmodel.AskViewModel
import org.koin.androidx.viewmodel.dsl.viewModel
import org.koin.dsl.module

val viewModelModule = module {
    viewModel {
        AskViewModel(
        )
    }
}