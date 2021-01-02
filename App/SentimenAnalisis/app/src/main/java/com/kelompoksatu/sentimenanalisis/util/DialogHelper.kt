package com.kelompoksatu.sentimenanalisis.util

import android.app.ProgressDialog
import android.content.Context


class CustomProgressDialog(context: Context, message: String) {
    var dialog: ProgressDialog = ProgressDialog(context)

    init {
        dialog.setMessage(message)
    }

    fun show() {
        dialog.setCancelable(false)
        dialog.show()
    }

    fun dismiss() {
        dialog.dismiss()
    }
}