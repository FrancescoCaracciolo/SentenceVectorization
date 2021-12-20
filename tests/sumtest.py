from SentenceVectorization import SentVE

SV = SentVE()
SV.load_model_from_file("models/glove.6B.50d.txt")

print(SV.sum_vectors_from_list([SV.vectors["king"], SV.vectors["queen"], SV.vectors["potato"], SV.vectors["chess"]]))
