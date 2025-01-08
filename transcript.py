import whisper_at as whisper

audio_tagging_time_resolution = 10
model = whisper.load_model("base")
result = model.transcribe("audio.mp3", at_time_res=audio_tagging_time_resolution)

# ASR Results
#print(result["text"])

# Audio Tagging Results
audio_tag_result = whisper.parse_at_label(result, 
language='follow_asr', 
top_k=1,  # only care of 1 single class
p_threshold=-5, # please tune this param
include_class_list=[16]) # 16 is the index of Laughter.

transciptntag = open("transcript.txt","w")
transciptntag.write(result["text"])

for record in audio_tag_result:
    if record["audio tags"] != []:
        transciptntag.write(record["audio tags"][0]+":  "+record['time']['start']+"-"+record[0]['time']['end'])
