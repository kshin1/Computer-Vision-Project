import numpy as np

emotion_dist = {}
emotion_list = ["h", "sa", "a", "n", "su", "c", "o"]
dist = ["dist_l", "dist_r", "dist_eyes", "dist_mouth_nose", "hght_mouth", "wdth_mouth"]
for e in emotion_list:
	temp_d = {}
	for d in dist: 
		temp_d[d] = []
	emotion_dist[e] = temp_d

with open("emotions.csv", "r") as f:
	count = 0
	for line in f.readlines():
		count += 1
		# Skip the headers
		if count == 1:
			continue
		line = line.rstrip()
		#emotion, dist_l, dist_r, dist_eyes, dist_mouth_nose, hght_mouth, wdth_mouth 
		row = line.split(",")
		
		for i,r in enumerate(row[1:]):
			emotion_dist[row[0]][dist[i]].append(float(r))

# Calculate the mean and standard deviation
emotion_mean_std = {}
for e, value in emotion_dist.items(): 
	emotion_mean_std[e] = {}
	for dist_name, d in value.items():
		mean = sum(d)/ len(d)
		std = np.std(np.array(d))
		emotion_mean_std[e][dist_name] = {"mean": mean, "std": std}

f1 = open("normalized_emotions.csv", "w")
f1.write("emotion,dist_l,dist_r,dist_eyes,dist_mouth_nose,hght_mouth,wdth_mouth\n")
for e, value in emotion_dist.items():
	e_img_len = len(emotion_dist[e]["dist_l"])
	for i in range(e_img_len):
		f1.write("{},{},{},{},{},{},{}\n".format(e, ",".join([str((emotion_dist[e][d][i] - emotion_mean_std[e][d]["mean"] ) / emotion_mean_std[e][d]["std"] )for d in dist])))
f1.close()
