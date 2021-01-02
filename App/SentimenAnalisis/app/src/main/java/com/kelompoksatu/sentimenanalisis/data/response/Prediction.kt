package com.kelompoksatu.sentimenanalisis.data.response


import com.google.gson.annotations.SerializedName

data class Prediction(
    @SerializedName("negatif")
    var negatif: Double,
    @SerializedName("netral")
    var netral: Double,
    @SerializedName("positif")
    var positif: Double,
    @SerializedName("prediction")
    var prediction: String,
    @SerializedName("removed_words")
    var removedWords: List<String>,
    @SerializedName("stopwords")
    var stopwords: List<String>,
    @SerializedName("used_terms")
    var usedTerms: List<String>
)