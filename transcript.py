import whisper_at as whisper

audio_tagging_time_resolution = 10
model = whisper.load_model("base")
result = model.transcribe("audio.mp3", at_time_res=audio_tagging_time_resolution)

# ASR Results
#print(result["text"])

# Audio Tagging Results
audio_tag_result = whisper.parse_at_label(result, 
language='follow_asr', 
top_k=13,  # only care of 1 single class
p_threshold=0, # please tune this param
include_class_list=[16,17,18,19,20,21,42,63,66,67,68,69,70]) # 16 is the index of Laughter.

transciptntag = open("transcript0.txt","w")
#transciptntag.write(result["text"])

#print(audio_tag_result)

for record in audio_tag_result:
    if record["audio tags"] != []:
        transciptntag.write(str(record["audio tags"][0])+":  "+str(record['time']['start'])+"-"+str(record['time']['end'])+"\n")
