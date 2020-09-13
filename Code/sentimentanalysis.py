import pandas as pd
from sklearn.model_selection import train_test_split
from tbrs import TermBasedRandomSampling
from preprocessing import Preprocessing
from naivebayes import NaiveBayes
from weighting import Weighting

# data = []

# data.append("Aku selama kuliah online benar-benar tidak belajar sama sekali. Ujian selalu tidak jujur, tugas tinggal memindahkan dari internet, dosen hanya memberi tugas, tidak pernah ada penjelasan materi. Ditambah semester 5 mau tetap daring, mau jadi apa Aku")
# data.append("Rasanya mau berhenti kuliah saja kalau daring begini, seperti bayar cuma cuma, materi dikasih secara online, disuruh baca sendiri tanpa ada yang menjelaskan, berasa otodidak :\"")
# data.append("Maaf, aku kuliah daring semakin malas. Kelas online saja ketiduran. Baik darimananya coba? Nilai sempurna bukan karena kita yang cerdas tapi karena dosennya yang kasihan sama kitanya. Tapi secara pemahaman, kosong sekali otak ini. Terima kasih")
# data.append("Sejujurnya aku oke-oke saja dengan kuliah daring. Cuma ya itu, kangen sama suasana kelas. Kalau corona sudah selesai, perpaduan offline-online sepertinya asik....")
# data.append("Ada yang mempeributkan masalah kuliah online/daring, sebagian ada yang menyalahkan dosen ada juga yang menyalahkan diri sendiri. Mau kuliah online atau tidak semua tergantung pribadi masing-masing dalam memahami materi yang dikasih dosen:)")
# data.append("Pak ini gimana anak sekolahan offline untuk beberapa zona, tapi kenapa mahasiswa tetap melaksanakan kuliah secara online / daring, justru mahasiswa lebih bisa beradaptasi dengan new normal dibandingkan dengan anak-anak yang masih sangat rentan, mohon dikaji lagi pak")
# data.append("Saya berdoa kuliah tetap daring saja, kampus mau offline padahal tempat masih zona merah dan kerabat aku yang kerjanya dokter saja suka bilang lagi kerja keras karena pasien corona. Lebih nyaman online, tetap dirumah adalah jalanku")
# data.append("Nilai positif saja yang diambil buang yang negatif. Positifnya (mungkin) ada beberapa mahasiswa yang tidak berani bertanya di kelas jadi lebih aktif bertanya di kuliah online (daring)")
# data.append("Benar juga ya lama lama kuliah online jadi new normal sampai masa pandemi ini selesai juga bisa jadi online, kalau bisa daring kenapa harus kuliah offline")

data = pd.read_excel(
    r'skripsi.xlsx',"Data Manualisasi")
data_tweet = data['Tweet']
data_target = data['Klasifikasi']

prepro = Preprocessing()
# cleaned_data, terms = prepro.preprocessing(data_tweet)

# weight = Weighting(cleaned_data, terms)
# tfidf = weight.get_tf_idf_weighting()

# tbrs = TermBasedRandomSampling(L=20,Y=100)
# stopwords = tbrs.create_stopwords(cleaned_data,terms)
stopwords = ['kuliah','online','daring','yang','saja','tetap','mau','offline','aku','materi','ada','ini','kelas','cara','sama','jadi','kasih','tapi','lebih','kalau','dosen','beberapa','kenapa','new','normal','zona','lagi','sendiri','dengan','masih','jujur','corona','paham','sekali','nilai','bisa','ya','selesai','jelas','mohon','kaji']			
print(stopwords)
# new_cleaned_data, new_terms = prepro.remove_stopword(cleaned_data,stopwords)
new_cleaned_data, new_terms = prepro.preprocessing(data_tweet,stopwords)
print(new_cleaned_data)

weight = Weighting(new_cleaned_data, new_terms)
# tfidf = weight.get_tf_idf_weighting()
# logtf = weight.get_log_tf_weighting()
rawtf = weight.get_raw_tf_weighting()
nb = NaiveBayes()

nb.fit(new_cleaned_data,new_terms,rawtf,data_target)
nb.predict(data_tweet[1],"Negatif")
# nb.predict(data_tweet[4],"Netral")
# nb.predict(data_tweet[7],"Positif")
# nb.predict("Saya bisa paham tapi kadang suka lupa","Netral")

