package com.kelompoksatu.sentimenanalisis.util


object Utils {

    fun printList(listString: List<String>): String {
        var output = ""
        var index = 0
        for (data in listString) {
            if (index != 0) {
                output = "$output, $data"
            } else {
                output = data
            }
            index++
        }
        return output
    }

    fun convertPercentageToInformation(num: Double): String {
        if (num > 90){
            return "Pasti"
        } else if (num > 80){
            return "Hampir Pasti"
        } else if (num > 60){
            return "Kemungkinan Besar"
        } else if (num > 40){
            return "Mungkin"
        } else {
            return "Tidak"
        }
    }

}