# coding: utf-8

import numpy as np

import os, sys
import cv2

import json

import psycopg2 as pg2
from psycopg2 import sql

import matplotlib.pyplot as plt


def load_configuration_file( filename ):
	with open( filename ) as data_file:
		data = json.load(data_file)

	return data


def get_uniform_distribution(low, high, number_of_samples):
	if number_of_samples != None:
		return np.random.uniform(low, high, number_of_samples)
	else:
		return np.random.uniform(low, high)


def get_learning_rates(low, high, number_of_samples):
	return 10 ** get_uniform_distribution(low, high, number_of_samples)

def get_weight_decays(low, high, number_of_samples):
	return 10 ** get_uniform_distribution(low, high, number_of_samples)

def get_adam_betas(low, high, number_of_samples):
	betas = []
	for i in range(0, number_of_samples):
		beta1 = 1
		beta2 = 0
		while ( beta1 > beta2 ):
			beta1 = 10 ** get_uniform_distribution(low, high, None)
			beta2 = 10 ** get_uniform_distribution(low, high, None)
			if beta1 < beta2:
				betas.append( (beta1, beta2) )

	return betas

def check_images_DB( sourcepath, schema_name, city_table_name, dbname, user, imgtype='Street', camera_views=['0', '90', '180', '270'], ext='.jpg', set_visited_no=False ):
	# Check points in POSTGRES database and flag OK if all images are readable
	# Table format: id, lat, lon, status, flg_visited
	# db_point_list format (use \COPY in POSTGRES): id, lat, lon

	#connect db postgres
	conn = pg2.connect(dbname=dbname, user=user)
	cur = conn.cursor()

	print(conn)
	print(cur)

	cur.execute( sql.SQL("SELECT id, lat, lon, flg_visited FROM {}.{} WHERE status != 'OK' ORDER BY id").format( sql.Identifier(schema_name), sql.Identifier(city_table_name) ) )
	result = cur.fetchall()

	print('Schema', schema_name, 'Table ', city_table_name, 'Total records', len(result))

	count = 0
	for record in result:

		_id = record[0]
		lat = record[1]
		lon = record[2]
		varstatus = record[3]

		camera_error = []

		if imgtype == 'Street':
			stat_ok = "Street_OK"
			stat_error = "Street_Error_"

		elif imgtype == 'Sat':
			stat_ok = "Sat_OK"
			stat_error = "Sat_Error_"

		else:
			stat_ok = "OK"
			stat_error = "Error_"

		for c in camera_views:
			nome_arquivo = sourcepath + "/" + str(lat) + "_" + str(lon) + "_" + str(c) + ext
			try:
				img = cv2.imread(nome_arquivo)
				if len(img.shape) != 3 or img.shape[2] != 3 or os.stat(nome_arquivo).st_size < 17000:
					camera_error.append(c)

			except:
				camera_error.append(c)


		if len(camera_error) == 0:
			#OK status for all cameraviews
			cur.execute( sql.SQL("UPDATE {}.{} SET status = %(stat)s WHERE id = %(id)s").format( sql.Identifier(schema_name), sql.Identifier(city_table_name) ), {'stat': varstatus + " " + stat_ok, 'id': _id})
			conn.commit()

		else:
			#Some cameras are damaged
			cur.execute( sql.SQL("UPDATE {}.{} SET status = %(stat)s WHERE id = %(id)s").format( sql.Identifier(schema_name), sql.Identifier(city_table_name) ), {'stat': varstatus + " " + stat_error + str(camera_error), 'id': _id})
			conn.commit()

			if set_visited_no == True:
				cur.execute( sql.SQL("UPDATE {}.{} SET flg_visited = %(visited)s WHERE id = %(id)s").format( sql.Identifier(schema_name), sql.Identifier(city_table_name) ), {'visited': "N", 'id': _id})
				conn.commit()



		count+=1
		sys.stdout.write("Progress/Total: %d/%d   \r" % (count, len(result)))
		sys.stdout.flush()

	print("Status...")

	cur.execute( sql.SQL("SELECT count(*) FROM {}.{} WHERE status ilike '%OK%'").format( sql.Identifier(schema_name), sql.Identifier(city_table_name) ) )
	result = cur.fetchall()
	print("Total OK points: ", len(result))

	cur.execute( sql.SQL("SELECT count(*) FROM {}.{} WHERE status not ilike '%OK%'").format( sql.Identifier(schema_name), sql.Identifier(city_table_name) ) )
	result = cur.fetchall()
	print("Total ERROR points: ", len(result))


	cur.close()
	conn.close()


# Plot distribution of citypointslist
def plot_pointlist_distribution(ciytpointlist):
	fr = open(ciytpointlist, 'r')
	points = fr.readlines()

	print(len(points))

	attributes = []
	for p in points:
		p = p.replace('\n', '')
		lat, lon, attr = p.split(';')

		attributes.append(attr)

	ord_attributes = sorted(attributes)

	plt.plot(ord_attributes, 'o')
	plt.show()


def imshow(img, ax):
	npimg = img.numpy()
	ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
	#ax.imshow(cv2.cvtColor(np.transpose(npimg, (1, 2, 0)), cv2.COLOR_RGB2BGR))
	ax.imshow(np.transpose(npimg, (1, 2, 0)))

#Function designed to plot prediction results in the right side of street images.
def imshow_info(img, ax, text_info, pred_info, facecolor='wheat', pos=(915,90)):
	print(img.shape)
	npimg = img.numpy()
	props_info = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	props_pred = dict(boxstyle='round', facecolor=facecolor, alpha=0.5)
	ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
	#ax.imshow(cv2.cvtColor(np.transpose(npimg, (1, 2, 0)), cv2.COLOR_RGB2BGR))
	ax.imshow(np.transpose(npimg, (1, 2, 0)))
	ax.text(pos[0], pos[1], text_info, fontsize=10, bbox=props_info)
	ax.text(pos[0], pos[1] + 50, pred_info, fontsize=10, bbox=props_pred)

def tensor_imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    inp = np.clip(inp, -1, 1)
    plt.imshow(inp)
    plt.show()
