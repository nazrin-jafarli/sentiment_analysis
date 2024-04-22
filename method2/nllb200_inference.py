from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M", token=True, src_lang="azj_Latn"
)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", token=True)

# sentence = "Güclü parol siyasətləri yaratmaq həssas məlumatların qorunmasında əsas rol oynayır."

def translate_with_nnlb200(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=30
    )
    translation=tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translation