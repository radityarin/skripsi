package com.kelompoksatu.sentimenanalisis.ui.diagnosis.viewmodel

import android.content.Context
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.kelompoksatu.sentimenanalisis.util.Constant
import kotlinx.coroutines.launch

class AskViewModel : ViewModel() {

    var output: MutableLiveData<String> = MutableLiveData()

    fun diagnosa(type: String, inputTweet: String, context: Context) {
        viewModelScope.launch {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(context))
            }
            val py = Python.getInstance()
            when (type) {
                Constant.DEMO1 -> {
                    val pyf = py.getModule(Constant.DEMO1+"pickle")
                    val obj = pyf.callAttr("predict", inputTweet)
                    output.postValue(obj.toString())
                }
                Constant.DEMO2 -> {
                    val pyf = py.getModule(Constant.DEMO2+"pickle")
                    val obj =
                        pyf.callAttr("predict", inputTweet)
                    output.postValue(obj.toString())
                }
            }
        }
    }


}