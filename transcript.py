import whisper_at as whisper

audio_tagging_time_resolution = 0.4
model = whisper.load_model("base")
result = model.transcribe("bbs01e02.mp3", at_time_res=audio_tagging_time_resolution)

# ASR Results
#print(result["text"])

# Audio Tagging Results
audio_tag_result = whisper.parse_at_label(result, 
language='follow_asr', 
top_k=13,  # only care of 1 single class
p_threshold=-5, # please tune this param
include_class_list=[16,17,18,19,20,21,42,63,66,67,68,69,70]) # 16 is the index of Laughter.

transciptntag = open("bbs01e01transcript-50.4.txt","w")
#transciptntag.write(result["text"])

#print(audio_tag_result)

for record in audio_tag_result:
    if record["audio tags"] != []:
        transciptntag.write(str(record["audio tags"][0])+":  "+str(record['time']['start'])+"-"+str(record['time']['end'])+"\n")
