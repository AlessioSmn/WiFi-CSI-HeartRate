import tensorflow as tf

model = tf.keras.models.load_model("models/csi_hr_best_200.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# abilita operazioni select TF (necessario per LSTM)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,    # tutte le ops TFLite standard
    tf.lite.OpsSet.SELECT_TF_OPS       # ops TF “extra”
]

# disabilita lowering delle TensorList ops (richiesto)
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()


# salva il modello convertito
with open("models/csi_hr_best_200.tflite", "wb") as f:
    f.write(tflite_model)
