package com.kelompoksatu.sentimenanalisis.ui.diagnosis.views

import android.graphics.Color
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.view.View
import androidx.lifecycle.Observer
import com.google.gson.Gson
import com.kelompoksatu.sentimenanalisis.R
import com.kelompoksatu.sentimenanalisis.data.response.Prediction
import com.kelompoksatu.sentimenanalisis.data.response.PredictionResponse
import com.kelompoksatu.sentimenanalisis.ui.diagnosis.viewmodel.AskViewModel
import com.kelompoksatu.sentimenanalisis.util.Constant
import com.kelompoksatu.sentimenanalisis.util.CustomProgressDialog
import com.kelompoksatu.sentimenanalisis.util.Utils
import kotlinx.android.synthetic.main.activity_main.*
import org.koin.androidx.viewmodel.ext.android.viewModel

class MainActivity : AppCompatActivity() {

    private val viewModel by viewModel<AskViewModel>()
    private lateinit var handler: Handler
    private lateinit var progressDialog: CustomProgressDialog

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        handler = Handler()

        initView()
        observeLiveData()

    }

    private fun initView() {
        progressDialog = CustomProgressDialog(
            this,
            getString(R.string.processing)
        )

        buttonPredict.setOnClickListener {
            progressDialog.show()
            diagnose(Constant.DEMO1, etInput.text.toString())
        }
    }

    private fun observeLiveData() {
        viewModel.output.observe(this, Observer {
            val predictionResponse: PredictionResponse =
                Gson().fromJson(it, PredictionResponse::class.java)
            onDataLoaded(predictionResponse)
        })
    }

    private fun onDataLoaded(predictionResponse: PredictionResponse) {
        val prediction: Prediction = predictionResponse.prediction
        when (predictionResponse.type){
            Constant.TBRS->{
                tvKelasTBRS.text = prediction.prediction
                tvNegatifTBRS.text = prediction.negatif.toString()
                tvNetralTBRS.text = prediction.netral.toString()
                tvPositifTBRS.text = prediction.positif.toString()
                tvStopwordsTBRS.text = Utils.printList(prediction.stopwords)
                tvFilteredWordsTBRS.text = Utils.printList(prediction.removedWords)
                tvUsedWordsTBRS.text = Utils.printList(prediction.usedTerms)
                diagnose(Constant.DEMO2,etInput.text.toString())
            }
            Constant.TALA->{
                progressDialog.dismiss()
                mcvResult.visibility = View.VISIBLE
                tvKelasTala.text = prediction.prediction
                tvNegatifTala.text = prediction.negatif.toString()
                tvNetralTala.text = prediction.netral.toString()
                tvPositifTala.text = prediction.positif.toString()
                tvStopwordsTala.text = Utils.printList(prediction.stopwords)
                tvStopwordsTala.setShowingLine(4)
                tvStopwordsTala.addShowMoreText("More")
                tvStopwordsTala.addShowLessText("Less")
                tvStopwordsTala.setShowMoreColor(Color.BLACK)
                tvStopwordsTala.setShowLessTextColor(Color.BLACK)
                tvFilteredWordsTala.text = Utils.printList(prediction.removedWords)
                tvFilteredWordsTala.setShowingLine(4)
                tvFilteredWordsTala.addShowMoreText("More")
                tvFilteredWordsTala.addShowLessText("Less")
                tvFilteredWordsTala.setShowMoreColor(Color.BLACK)
                tvFilteredWordsTala.setShowLessTextColor(Color.BLACK)
                tvUsedWordsTala.text = Utils.printList(prediction.usedTerms)
            }
        }
    }

    private fun diagnose(type: String, inputTweet: String) {
        handler.postDelayed({
            viewModel.diagnosa(type, inputTweet, applicationContext)
        }, 500)

    }
}