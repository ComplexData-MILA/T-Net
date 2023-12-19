'''
CAUTION: There are explicit words used in the code as keywords to flag certain behaviour. 
Please exercise caution while reading.

Defining labeling functions for M.O/cluster classification.
The classes under consideration are "Spa", "Spam", "HT", "ISW" (Independent Sex Worker)

This code also outputs the distribution plots of the labeling functions in the results/ folder.
Additionally, the spatial distribution of the HT based LFs are also analysed and added to the results/ folder

Usage: python labeling_functions.py
Author: Pratheeksha Nair
'''

import pandas as pd
import numpy as np
import haversine as hs
import argparse

from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
import pickle as pkl
import ast
import matplotlib.pyplot as plt
from tqdm import tqdm

## HT Indicators

THIRD_PERSON = ["she", "her", "hers", "herself"] # Third person pronouns.
FIRST_PERSON_PLURAL = ["us", "our", "ours", "ourselves"] # First person plural pronouns

PRONOUNS = THIRD_PERSON + FIRST_PERSON_PLURAL

NON_RESTRICTED_SEX_SEQUENCE = ["bb", "raw", "bareback", "bare back", "no cover", "without condom", "no condom"]
NON_RESTRICTED_SEX_SEQUENCE += ['BBBJ',"BBBj",'bxxJ','b88j','B-B-B-J']

INCALL_ONLY = ["in call", "in call only", "in-call only", "incall only"]
NO_OUTCALL = ["no outcall", "no out call", "no out-call"]

HT_KEYWORDS = ["new in town", "new girl", "came to the town", "out off town", 
			   "few days",  "back to town", "out of town", "in your town", 
			   "for a few days", "back in town",  "only stay for a few", "new arrived", 
			   "just arrived", "new here", "new to this",  "new agency", "new in town", 
			   "new to your city", "i'm new", "i am new", "brand new",  "little new",
			   "very new", "look new", "short-term", "new to the area", "new to the industry", 
			   "new in the business", "new face", "short time", "in town for", "new to town", 
			   "short stay", "short term", "short time", "short period", "arrived in town", 
			   "arrived short term", "for a limited time", "newly arrived", "passing through", 
			   "on vacation", "in town for a few days", "i am a newb", "i am a newbie", "first time to", 
			   "first time in", "first time here", "first time visiting", "new in the field",
			   "just turned 18", "turned 18", "hot teen", "fresh", "petite", "student", "college", "young",
			   "juicy", "tight"]


## ISW Indicators
WITH_CONDOM_SEQUENCE = ["with condom", "use of condoms", "with a condom", "no bb", "safe", "safe play", "covered"]
RESTRICTED_SEX_SEQUENCE = ["no oral", "no anal", "no black", "no greek", "gentlemen only", "respectful"]
RESTRICTED_SEX_SEQUENCE += WITH_CONDOM_SEQUENCE

INCALL_WORDS = ["incall", "in-call", "incalls", 'in call']
OUTCALL_WORDS = ["outcall", "out-call", "outcalls", 'out call']
CARCALL_WORDS = ["carcall", "car-call", "cardate", 'carplay']
NO_INCALL_SEQUENCE = ["no incalls", "no incall", "no in-calls", "no in calls", "no incall"]
OUTCALL_SEQUENCE = ["out call", "out call only", "out-call only", "outcall only"]

SW_INCALL_WORDS = INCALL_WORDS + OUTCALL_SEQUENCE + CARCALL_WORDS + NO_INCALL_SEQUENCE

SW_KEYWORDS = ["mature", "classy", "real", "deposit", 'advance', "professional", "appointment", 'milf', 'mommy']


# def ht_lf_analysis(df):
	# function to look at which locations have been flagged with high values of HT indicators

def find_loc_radii(list_xcoords, list_ycoords):
	# list_locs = list_locs.apply(lambda x: x.split())
	list_locs = list(zip(list_xcoords, list_ycoords))
	
	t_lat = sorted(list_locs, key=lambda x: float(x[0]),reverse=True)
	t_lon = sorted(list_locs, key=lambda x: float(x[1]),reverse=True)

	x11 = float(t_lat[0][0])
	y11 = float(t_lat[0][1])
	x21 = float(t_lat[-1][0])
	y21 = float(t_lat[-1][1])
	
	x12 = float(t_lon[0][0])
	y12 = float(t_lat[0][1])
	x22 = float(t_lat[-1][0])
	y22 = float(t_lat[-1][1])
	
	d1 = hs.haversine((x11,y11),(x21,y21))
	d2 = hs.haversine((x12,y12),(x22,y22))
	# print((x11,y11),(x21,y21), d1)
	# print((x12,y12),(x22,y22), d1)
	# print()
	# exit()
	radius = max(d1, d2)
	return radius

## Massage parlor indicators
def check_massage_parlour(txt):
	if 'spa' in txt.lower():
		return len(np.where(np.array(txt.split())=='spa')[0])
	else:
		return 0

def apply_lfs(df, level_of_analysis='Meta label'):
	ht_keyword_map = defaultdict(float)
	non_restricted_map = defaultdict(float)
	incall_map = defaultdict(float)
	pronoun_map = defaultdict(float)
	sw_keyword_map = defaultdict(float)
	restricted_map = defaultdict(float)
	sw_incall_map = defaultdict(float)

	num_people_map = defaultdict(int)
	spa_count_map = defaultdict(float)
	loc_radius_over_time = defaultdict(float)
	num_imgs_per_phone = defaultdict(float)
	num_locs = defaultdict(float)
	num_phones = defaultdict(float)

	spa_names = pkl.load(open("marinus_labelled/spa_names.pkl",'rb'))

	if 'location' not in df.columns:
		df = df.rename(columns={'city_id':'location'})
	spa_ads = []
	for id, row in tqdm(df.iterrows()):
		if row.location not in spa_names.keys():
			continue
		for s in spa_names[row.location]:
			if s.lower() in row.description.lower() and s.lower() not in ['spa','soul']:
				spa_ads.append(id)
	spa_flag = np.array([None]*len(df))
	spa_flag[spa_ads] = True
	df['in_spa_list'] = spa_flag

	for clus_id, grp in tqdm(df.groupby(level_of_analysis)):
		loc_radius = find_loc_radii(grp.xcoord.values, grp.ycoord.values)
		# dates = pd.to_datetime(grp['date_posted'], infer_datetime_format=True)
		dates = pd.to_datetime(grp['post_date'], infer_datetime_format=True)
		dates = sorted(dates)
		diff = (dates[-1]-dates[0])/np.timedelta64(1,'D')

		if diff == 0:
			val = loc_radius/10e-3
		else:
			val = loc_radius/diff
		if grp.phone_num.nunique() != 0:
			img_cnt = grp.image_id.nunique()/grp.phone_num.nunique()
		else:
			img_cnt = 0
		
		if 'NEAT_names' in grp.columns:
			new_names = []
			for item in grp.NEAT_names.unique():
				new_names.extend(list(set(list(ast.literal_eval(item)))))
			new_names = list(set(list(map(lambda x: x.lower(), new_names))))

			num_people_map[clus_id] += len(np.unique(new_names))
		else:
			num_people_map[clus_id] += 0

		grp['spa_keyword'] = grp.description.apply(lambda x:check_massage_parlour(x))		
		
		if grp['spa_keyword'].unique()[0] > 0:
			spa_count_map[clus_id] += grp['spa_keyword'].unique()[0]
		elif grp['in_spa_list'].unique()[0]:
			spa_count_map[clus_id] += 100
		
		loc_radius_over_time[clus_id] += val
		num_imgs_per_phone[clus_id] += img_cnt
		num_locs[clus_id] += grp['location'].nunique()
		num_phones[clus_id] += grp['phone_num'].nunique()
		
		for txt in grp.description.unique():
			ad_words = txt.lower().split()
			for word in HT_KEYWORDS:
				if word in txt:
					ht_keyword_map[clus_id] += 1
			for word in SW_KEYWORDS:
				if word in txt:
					sw_keyword_map[clus_id] += 1
			for word in NON_RESTRICTED_SEX_SEQUENCE:
				if word in txt:
					non_restricted_map[clus_id] += 1
			for word in RESTRICTED_SEX_SEQUENCE:
				if word in txt:
					restricted_map[clus_id] += 1
			for word in SW_INCALL_WORDS:
				if word in txt:
					sw_incall_map[clus_id] += 1

			# if len(set(ad_words) & set(INCALL_WORDS) & set(OUTCALL_WORDS) & set(CARCALL_WORDS)):
			# 	incall_map[clus_id] += 1
			# else:
			for word in INCALL_ONLY:
				if word in txt:
					incall_map[clus_id] += 1
			for word in NO_OUTCALL:
				if word in txt:
					incall_map[clus_id] += 1
			for word in INCALL_WORDS:
				if word in txt:
					incall_map[clus_id] += 1
			for word in OUTCALL_WORDS:
				if word in txt:
					incall_map[clus_id] += 1
			for word in CARCALL_WORDS:
				if word in txt:
					incall_map[clus_id] += 1
			for word in PRONOUNS:
				if word in ad_words:
					pronoun_map[clus_id] += 1

		ht_keyword_map[clus_id] /= grp.ad_id.count()*100
		non_restricted_map[clus_id] /= grp.ad_id.count()*100
		incall_map[clus_id] /= grp.ad_id.count()*100
		pronoun_map[clus_id] /= grp.ad_id.count()*100
		sw_keyword_map[clus_id] /= grp.ad_id.count()*100
		restricted_map[clus_id] /= grp.ad_id.count()*100
		sw_incall_map[clus_id] /= grp.ad_id.count()*100


	sorted_kw_map = {}
	sorted_non_rest_map = {}
	sorted_incall_map = {}
	sorted_pro_map = {}
	sorted_rest_map = {}
	sorted_sw_map = {}
	sorted_sw_incall_map = {}

	sorted_num_people_map = {}
	sorted_spa_count_map = {}
	sorted_loc_radius_over_time = {}
	sorted_num_imgs_per_phone ={}
	sorted_num_locs = {}
	sorted_num_phones = {}

	for item in sorted(ht_keyword_map.items(),key=lambda x:x[1]):
			sorted_kw_map[item[0]] = item[1]
	for item in sorted(non_restricted_map.items(),key=lambda x:x[1]):
		if item[1] > 0:
			sorted_non_rest_map[item[0]] = item[1]
			
	for item in sorted(restricted_map.items(),key=lambda x:x[1]):
		if item[1] > 0:
			sorted_rest_map[item[0]] = item[1]
			
	for item in sorted(sw_keyword_map.items(),key=lambda x:x[1]):
		if item[1] > 0:
			sorted_sw_map[item[0]] = item[1]

	for item in sorted(sw_incall_map.items(),key=lambda x:x[1]):
		if item[1] > 0:
			sorted_sw_incall_map[item[0]] = item[1]
			
	for item in sorted(incall_map.items(),key=lambda x:x[1]):
		if item[1] > 0:
			sorted_incall_map[item[0]] = item[1]
			
	for item in sorted(pronoun_map.items(),key=lambda x:x[1]):
		if item[1] > 0:
			sorted_pro_map[item[0]] = item[1]
			
	for item in sorted(num_people_map.items(),key=lambda x:x[1]):
		if item[1] > 0:
			sorted_num_people_map[item[0]] = item[1]
			
	for item in sorted(spa_count_map.items(),key=lambda x:x[1]):
		if item[1] > 0:
			sorted_spa_count_map[item[0]] = item[1]
			
	for item in sorted(loc_radius_over_time.items(),key=lambda x:x[1]):
		if item[1] > 0:
			sorted_loc_radius_over_time[item[0]] = item[1]
			
	for item in sorted(num_imgs_per_phone.items(),key=lambda x:x[1]):
		if item[1] > 0:
			sorted_num_imgs_per_phone[item[0]] = item[1]
			
	for item in sorted(num_phones.items(),key=lambda x:x[1]):
		if item[1] > 0:
			sorted_num_phones[item[0]] = item[1]

	for item in sorted(num_locs.items(),key=lambda x:x[1]):
		if item[1] > 0:
			sorted_num_locs[item[0]] = item[1]

	# finding inflection points
	clusters1 = list(sorted_kw_map.keys())
	clusters2 = list(sorted_non_rest_map.keys())
	clusters3 = list(sorted_incall_map.keys())
	clusters4 = list(sorted_pro_map.keys())
	clusters5 = list(sorted_sw_map.keys())
	clusters6 = list(sorted_rest_map.keys())
	clusters7 = list(sorted_sw_incall_map.keys())

	clusters8 = list(sorted_num_people_map.keys())
	clusters9 = list(sorted_spa_count_map.keys())
	clusters10 = list(sorted_loc_radius_over_time.keys())
	clusters11 = list(sorted_num_imgs_per_phone.keys())
	clusters12 = list(sorted_num_locs.keys())
	clusters13 = list(sorted_num_phones.keys())

	percentages_kw = list(sorted_kw_map.values())
	percentages_non_rest = list(sorted_non_rest_map.values())
	percentages_incall = list(sorted_incall_map.values())
	percentages_pro = list(sorted_pro_map.values())
	percentages_sw = list(sorted_sw_map.values())
	percentages_rest = list(sorted_rest_map.values())
	percentages_sw_incall = list(sorted_sw_incall_map.values())

	perc_num_ppl = list(sorted_num_people_map.values())
	perc_spa_count = list(sorted_spa_count_map.values())
	perc_loc_radius_time = list(sorted_loc_radius_over_time.values())
	perc_num_img_per_phone = list(sorted_num_imgs_per_phone.values())
	perc_num_locs = list(sorted_num_locs.values())
	perc_num_phones = list(sorted_num_phones.values())

	sorted_clusters1 = [x for _,x in sorted(zip(percentages_kw, clusters1))]
	sorted_percentages_kw = [x for x,_ in sorted(zip(percentages_kw, clusters1))]
	sorted_clusters2 = [x for _,x in sorted(zip(percentages_non_rest, clusters2))]
	sorted_percentages_rest = [x for x,_ in zip(percentages_non_rest, clusters2)]
	sorted_clusters3 = [x for _,x in sorted(zip(percentages_incall, clusters3))]
	sorted_incall = [x for x,_ in zip(percentages_incall, clusters3)]
	sorted_clusters4 = [x for _,x in zip(percentages_pro, clusters4)]
	sorted_pro = [x for x,_ in zip(percentages_pro, clusters4)]

	sorted_clusters5 = [x for _,x in sorted(zip(percentages_sw, clusters5))]
	sorted_sw = [x for x,_ in zip(percentages_sw, clusters5)]
	sorted_clusters6 = [x for _,x in sorted(zip(percentages_rest, clusters6))]
	sorted_rest = [x for x,_ in zip(percentages_rest, clusters6)]
	sorted_clusters7 = [x for _,x in zip(percentages_sw_incall, clusters7)]
	sorted_sw_incall = [x for x,_ in zip(percentages_sw_incall, clusters7)]

	sorted_clusters8 = [x for _,x in sorted(zip(perc_num_ppl, clusters8))]
	sorted_num_ppl = [x for x,_ in zip(perc_num_ppl, clusters8)]
	sorted_clusters9 = [x for _,x in sorted(zip(perc_spa_count, clusters9))]
	sorted_spa = [x for x,_ in zip(perc_spa_count, clusters9)]
	sorted_clusters10 = [x for _,x in zip(perc_loc_radius_time, clusters10)]
	sorted_loc_radius = [x for x,_ in zip(perc_loc_radius_time, clusters10)]

	sorted_clusters11 = [x for _,x in sorted(zip(perc_num_img_per_phone, clusters11))]
	sorted_img = [x for x,_ in zip(perc_num_img_per_phone, clusters11)]
	sorted_clusters12 = [x for _,x in sorted(zip(perc_num_locs, clusters12))]
	sorted_locs = [x for x,_ in zip(perc_num_locs, clusters12)]
	sorted_clusters13 = [x for _,x in zip(perc_num_phones, clusters13)]
	sorted_phones = [x for x,_ in zip(perc_num_phones, clusters13)]


	inflections = []
	color_map = []
	color_set = ['r', 'r', 'r', 'r', 'g', 'g', 'g', 'r', 'k', 'b', 'r', 'b', 'b']
	for i, lst in enumerate([sorted_percentages_kw, sorted_percentages_rest, sorted_incall, sorted_pro, \
				sorted_sw, \
				sorted_rest, sorted_sw_incall, sorted_num_ppl, sorted_spa, sorted_loc_radius, sorted_img, \
				sorted_locs, sorted_phones]):
		# smooth
		smooth = gaussian_filter1d(lst, 100)
		# compute second derivative
		smooth_d2 = np.gradient(np.gradient(smooth))

		# if i == 6:
			# infls = np.argsort(abs(np.diff(smooth)))[0]
		if i == 9:
			infls = 900
		# 	print(np.where(np.argsort(np.diff(smooth)))[0])
		# 	# infls = np.argsort(abs(np.diff(smooth)))[0]
			# infls = 26
		# elif i == 5:
			# infls = 30
		elif i == 11:
			print(np.min(lst))
			zero_inds = np.where(lst==0)[0]
			if len(zero_inds) == 0:
				infls11 = None
			else:
				infls11 = min(zero_inds)
			# infls = 420
			infls = 1900
			print(infls11)
		elif i == 12:
		# 	print(np.min(lst))
			zero_inds = np.where(lst==0)[0]
			if len(zero_inds) == 0:
				infls12 = None
			else:
				infls12 = min(zero_inds)
			# infls = 410
			infls = 1500
		# 	print(infls12)
		elif i == 0:
			# infls = 400
			infls = 2400
		# elif i == 1:
			# infls = 110
		# elif i == 3:
			# infls = 80
		# elif i == 10:
			# infls = 400
		elif i == 7:
			# infls = 180
			infls = 1200
		else:
		# find switching points	
			try:
				infls = np.where(np.diff(np.sign(smooth_d2)))[0][-1]
				# infls = np.argsort(abs(np.diff(smooth)))[0]
			except:
				# print(smooth_d2, i, '2nd')
				t = np.where(np.diff(np.sign(smooth_d2)))[0]
				if i == 8:
					infls = 0
				else:
					infls = np.argsort(abs(np.diff(smooth)))[0]
					# infls = 0
		inflections.append(infls)
		colors = np.array(['tab:gray']*len(lst))
		colors[infls:] = np.array(color_set[i])
		color_map.append(colors)

	pkl.dump(color_map, open("plotting_code/ht_canada/color_map.pkl",'wb'))
	pkl.dump(inflections, open("plotting_code/ht_canada/inflections.pkl",'wb'))
	# pkl.dump([sorted_percentages_kw, sorted_percentages_rest, sorted_incall, sorted_pro, \
	# 			sorted_sw, sorted_rest, sorted_img, sorted_num_ppl, sorted_sw_incall, sorted_loc_radius, \
	# 			sorted_locs, sorted_phones], open("plotting_code/ht_canada/plot_values.pkl",'wb'))
	# pkl.dump([sorted_clusters1, sorted_clusters2, sorted_clusters3, sorted_clusters4, \
	# 			sorted_clusters5, sorted_clusters6, sorted_clusters11, sorted_clusters8, sorted_clusters7, sorted_clusters10, \
	# 			sorted_clusters12, sorted_clusters13], open("plotting_code/ht_canada/plot_keys.pkl",'wb'))	

	fig, (row1, row2, row3) = plt.subplots(3, 4, figsize=[45,20], constrained_layout=True)

	plt.rcParams.update({'font.size': 22, 'font.family':"Times New Roman"})

	ax1=row1[0]
	ax2=row1[1]
	ax3=row1[2]
	ax4=row1[3]
	ax5=row2[0]
	ax6=row2[1]
	ax7=row2[2]
	ax8=row2[3]
	ax9=row3[0]
	ax10=row3[1]
	ax11=row3[2]
	ax12=row3[3]
	# ax13=row4[0]

	ax1.scatter(x=range(len(sorted_clusters1)), y=sorted_percentages_kw, c=color_map[0])
	ax1.axvline(x=inflections[0], color='k')
	ax2.scatter(x=range(len(sorted_clusters2)), y=sorted_percentages_rest, c=color_map[1])
	ax2.axvline(x=inflections[1], color='k')
	ax3.scatter(x=range(len(sorted_clusters3)), y=sorted_incall, c=color_map[2])
	ax3.axvline(x=inflections[2], color='k')
	ax4.scatter(x=range(len(sorted_clusters4)), y=sorted_pro, c=color_map[3])
	ax4.axvline(x=inflections[3], color='k')
	ax5.scatter(x=range(len(sorted_clusters5)), y=sorted_sw, c=color_map[4])
	ax5.axvline(x=inflections[4], color='k')
	ax6.scatter(x=range(len(sorted_clusters6)), y=sorted_rest, c=color_map[5])
	ax6.axvline(x=inflections[5], color='k')
	ax7.scatter(x=range(len(sorted_clusters11)), y=sorted_img, c=color_map[10])
	ax7.axvline(x=inflections[10], color='k')

	ax8.scatter(x=range(len(sorted_clusters8)), y=sorted_num_ppl, c=color_map[7])
	ax8.axvline(x=inflections[7], color='k')
	ax9.scatter(x=range(len(sorted_clusters7)), y=sorted_sw_incall, c=color_map[6])
	ax9.axvline(x=inflections[6], color='k')
	# ax10.scatter(x=range(len(sorted_clusters9)), y=sorted_spa, label='Spa \nindicators')
	# ax10.axvline(x=inflections[8], color='k')
	ax10.scatter(x=range(len(sorted_clusters10)), y=sorted_loc_radius, c=color_map[9])
	ax10.axvline(x=inflections[9], color='k')
	
	ax11.scatter(x=range(len(sorted_clusters12)), y=sorted_locs, c=color_map[11])
	ax11.axvline(x=inflections[11], color='k')
	if infls11 != None:
		ax11.axvline(x=infls11, color='k')
	ax12.scatter(x=range(len(sorted_clusters13)), y=sorted_phones, c=color_map[12])
	ax12.axvline(x=inflections[12], color='k')
	if infls12 != None:
		ax11.axvline(x=infls12, color='k')

	# fig.supylabel("Value", fontsize = 18.0)
	fig.supxlabel("Micro-cluster ID", fontsize = 22.0)

	ax1.set_title('HT keywords', fontname='Times New Roman', fontsize=22)
	ax2.set_title('No restricted services', fontname='Times New Roman', fontsize=22)
	ax3.set_title('Incall only/\nNo outcall', fontname='Times New Roman', fontsize=22)
	ax4.set_title('3rd/1st person plural', fontname='Times New Roman', fontsize=22)
	ax5.set_title('Non HT keywords', fontname='Times New Roman', fontsize=22)
	ax6.set_title('Restricted services', fontname='Times New Roman', fontsize=22)
	ax7.set_title('Img per phone', fontname='Times New Roman', fontsize=22)

	ax8.set_title('Num persons', fontname='Times New Roman', fontsize=22)
	ax9.set_title('Availability', fontname='Times New Roman', fontsize=22)
	ax10.set_title('Loc radius over time', fontname='Times New Roman', fontsize=22)
	ax11.set_title('Num locations', fontname='Times New Roman', fontsize=22)
	ax12.set_title('Num phones', fontname='Times New Roman', fontsize=22)

	# ax1.legend(fontsize = 12.0)
	# ax2.legend(fontsize = 12.0)
	# ax3.legend(fontsize = 12.0)
	# ax4.legend(fontsize = 12.0)
	# ax5.legend(fontsize = 12.0)
	# ax6.legend(fontsize = 12.0)
	# ax7.legend(fontsize = 12.0)

	# ax8.legend(fontsize = 12.0)
	# ax9.legend(fontsize = 12.0)
	# ax10.legend(fontsize = 18.0)
	# ax11.legend(fontsize = 18.0)
	# ax12.legend(fontsize = 18.0)
	# ax13.legend(fontsize = 30.0)

	# row4[0].set_visible(False)
	# row4[1].set_visible(False)
	# row4[2].set_visible(False)
	# row4[3].set_visible(False)

	# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
	# plt.savefig("results/lf_dists_v2.png",bbox_inches='tight')
	plt.show()
	exit()
	

	kw_clusters = [c for c, v in sorted_kw_map.items() if v > sorted_percentages_kw[inflections[0]]]
	no_rest_clusters = [c for c, v in sorted_non_rest_map.items() if v > sorted_percentages_rest[inflections[1]]]
	incall_clusters = [c for c, v in sorted_incall_map.items() if v > sorted_incall[inflections[2]]]
	pro_clusters = [c for c, v in sorted_pro_map.items() if v > sorted_pro[inflections[3]]]
	sw_clusters = [c for c, v in sorted_sw_map.items() if v > sorted_sw[inflections[4]]]
	rest_clusters = [c for c, v in sorted_rest_map.items() if v > sorted_rest[inflections[5]]]
	sw_incall_clusters = [c for c, v in sorted_sw_incall_map.items() if v > sorted_sw_incall[inflections[6]]]
	num_ppl_clusters = [c for c, v in sorted_num_people_map.items() if v > sorted_num_ppl[inflections[7]]]
	loc_radius_clusters = [c for c, v in sorted_loc_radius_over_time.items() if v > sorted_loc_radius[inflections[9]]]
	img_clusters = [c for c, v in sorted_num_imgs_per_phone.items() if v > sorted_img[inflections[10]]]
	locs_clusters = [c for c, v in sorted_num_locs.items() if (v > sorted_locs[inflections[11]]) or (v == infls11)]
	phones_clusters = [c for c, v in sorted_num_phones.items() if (v > sorted_phones[inflections[12]]) or (v == infls12)]

	cluster_label = level_of_analysis
	df['kw_label'] = False
	df[df[cluster_label].isin(kw_clusters)]['kw_label'] = True


	df['no_rest_label'] = False
	df[df[cluster_label].isin(no_rest_clusters)]['no_rest_label'] = True


	df['incall_label'] = False
	df[df[cluster_label].isin(incall_clusters)]['incall_label'] = True


	df['pro_label'] = False
	df[df[cluster_label].isin(pro_clusters)]['pro_label'] = True


	df['sw_label'] = False
	df[df[cluster_label].isin(sw_clusters)]['sw_label'] = True


	df['rest_label'] = False
	df[df[cluster_label].isin(rest_clusters)]['rest_label'] = True


	df['sw_incall_label'] = False
	df[df[cluster_label].isin(sw_incall_clusters)]['sw_incall_label'] = True


	df['num_ppl_label'] = False
	df[df[cluster_label].isin(num_ppl_clusters)]['num_ppl_label'] = True


	df['loc_radius_label'] = False
	df[df[cluster_label].isin(loc_radius_clusters)]['loc_radius_label'] = True


	df['img_label'] = False
	df[df[cluster_label].isin(img_clusters)]['img_label'] = True


	df['locs_label'] = False
	df[df[cluster_label].isin(locs_clusters)]['locs_label'] = True


	df['phones_label'] = False
	df[df[cluster_label].isin(phones_clusters)]['phones_label'] = True


	spa_clusters = sorted_spa_count_map.keys()
	spa_clusters = []
	df_without_spa = df[~df[level_of_analysis].isin(spa_clusters)]
	df_with_labels = df_without_spa[[level_of_analysis]+['kw_label','no_rest_label','incall_label','pro_label','sw_label', 'rest_label',\
				'sw_incall_label','num_ppl_label','loc_radius_label','img_label','locs_label','phones_label']]    

	label_mat = np.zeros(shape=[df_with_labels[level_of_analysis].nunique(), 4])
	lf_votes = defaultdict(dict)
	i = 0
	for clus, grp in tqdm(df_with_labels.groupby(level_of_analysis)):
		lf_votes[clus]['ht'] = [clus in kw_clusters, clus in no_rest_clusters, clus in incall_clusters, \
						   clus in pro_clusters, clus in num_ppl_clusters, clus in img_clusters]
		ht_score = np.mean([clus in kw_clusters, clus in no_rest_clusters, clus in incall_clusters, \
						   clus in pro_clusters, clus in num_ppl_clusters, clus in img_clusters])


		lf_votes[clus]['isw'] = [clus in sw_clusters, clus in rest_clusters, clus in sw_incall_clusters]
		sw_score = np.mean([clus in sw_clusters, clus in rest_clusters, clus in sw_incall_clusters])


		lf_votes[clus]['spam'] = [clus in phones_clusters, clus in loc_radius_clusters, clus in locs_clusters, \
							  ]
		spam_score = np.mean([clus in phones_clusters, clus in loc_radius_clusters, clus in locs_clusters, \
							  ])

		# label_mat[clus] = [spam_score, ht_score, sw_score]
		label_mat[i][0] = clus
		label_mat[i][1] = spam_score
		label_mat[i][2] = ht_score
		label_mat[i][3] = sw_score
		i += 1
	
	return label_mat, list(spa_clusters), lf_votes
	# return sure_labels, ht_clusters, spam_clusters, isw_clusters, spa_clusters

def preprocess(df, cities):
	cities = cities[cities.country_id==3]
	df = pd.merge(df, cities, left_on='city_id', right_on='id')
	df['geolocation'] = str(df.xcoord) + " " + str(df.ycoord)
	df.rename(columns={'phone':'phone_num', 'body':'description'}, inplace=True)
	return df

def get_data_df():
	parser = argparse.ArgumentParser()
	parser.add_argument('--filename', help='Path of the file to be preprocessed', \
		default='data/HT2018_final_trimmed_for_labeling_neat_preprocessed.csv')
	parser.add_argument('--cities', help='Path to the cities file', default='data/cities.csv')
	parser.add_argument('--level_of_analysis', \
		help='What level of analysis is required', default='Meta label', choices=['Meta label', 'LSH label', 'ad_id'])
	args = parser.parse_args()
	data = pd.read_csv(args.filename, index_col=False)
	cities = pd.read_csv(args.cities, index_col=False)
	level_of_analysis = args.level_of_analysis

	print(level_of_analysis)
	print(data[level_of_analysis].nunique())

	if 'geolocation' not in data.columns:
		data = preprocess(data, cities)

	return data, cities, level_of_analysis



if __name__ == "__main__":

	data, cities, level_of_analysis = get_data_df()

	label_mat, spa_clusters, lf_votes = apply_lfs(data, level_of_analysis)

	# pkl.dump(label_mat, open("results/label_mat_sm.pkl",'wb'))
	# pkl.dump(spa_clusters, open("results/spa_clusters.pkl",'wb'))
	# pkl.dump(lf_votes, open("results/lf_votes_sm.pkl",'wb'))


