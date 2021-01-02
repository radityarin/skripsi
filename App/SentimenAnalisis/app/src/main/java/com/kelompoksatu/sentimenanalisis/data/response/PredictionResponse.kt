package com.kelompoksatu.sentimenanalisis.data.response


import com.google.gson.annotations.SerializedName

data class PredictionResponse(
    @SerializedName("data")
    var prediction: Prediction,
    @SerializedName("type")
    var type: String,
    @SerializedName("success")
    var success: Boolean
)