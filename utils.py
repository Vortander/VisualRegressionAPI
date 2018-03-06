# coding: utf-8

import numpy as np

import os, sys
import cv2

import psycopg2 as pg2
from psycopg2 import sql

import matplotlib.pyplot as plt


def check_images_DB( sourcepath, schema_name, city_table_name, dbname, user ):
	# Check points in POSTGRES database and flag OK if all images are readable
	# Table format: id, lat, lon, status, flg_visited
	# db_point_list format (use \COPY in POSTGRES): id, lat, lon

	#connect db postgres
	conn = pg2.connect(dbname=dbname, user=user)
	cur = conn.cursor()

	print(conn)
	print(cur)

	#cur.execute( sql.SQL("SELECT id, lat, lon, flg_visited FROM {}.{} WHERE status != 'OK' ORDER BY id").format( sql.Identifier(schema_name), sql.Identifier(city_table_name) ) )
	cur.execute( sql.SQL("SELECT id, lat, lon, flg_visited FROM {}.{} ORDER BY id").format( sql.Identifier(schema_name), sql.Identifier(city_table_name) ) )
	result = cur.fetchall()	
	
	print('Schema', schema_name, 'Table ', city_table_name, 'Total records', len(result))

	count = 0
	for record in result:

		_id = record[0]
		lat = record[1]
		lon = record[2]

		camera_error = []

		for c in ['0', '90', '180', '270']:
			nome_arquivo = sourcepath + "/" + str(lat) + "_" + str(lon) + "_" + str(c) + ".jpg"
			try:
				img = cv2.imread(nome_arquivo)
				if len(img.shape) != 3 or img.shape[2] != 3 or os.stat(nome_arquivo).st_size < 17000:
					camera_error.append(c)

			except:
				camera_error.append(c)


		if len(camera_error) == 0:
			#OK status for all cameraviews
			cur.execute( sql.SQL("UPDATE {}.{} SET status = %(stat)s WHERE id = %(id)s").format( sql.Identifier(schema_name), sql.Identifier(city_table_name) ), {'stat': 'OK', 'id': _id})
			conn.commit()

		else:
			#Some cameras are damaged
			cur.execute( sql.SQL("UPDATE {}.{} SET status = %(stat)s WHERE id = %(id)s").format( sql.Identifier(schema_name), sql.Identifier(city_table_name) ), {'stat': 'Error_' + str(camera_error), 'id': _id})
			conn.commit()

		count+=1
		sys.stdout.write("Progress/Total: %d/%d   \r" % (count, len(result)))
		sys.stdout.flush()

	print("Status...")

	cur.execute( sql.SQL("SELECT count(*) FROM {}.{} WHERE status = 'OK'").format( sql.Identifier(schema_name), sql.Identifier(city_table_name) ) )
	result = cur.fetchall()
	print("Total OK points: ", len(result))

	cur.execute( sql.SQL("SELECT count(*) FROM {}.{} WHERE status != 'OK'").format( sql.Identifier(schema_name), sql.Identifier(city_table_name) ) )
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


def imshow(img):
	print(img.shape)
	npimg = img.numpy()
	plt.axis("off")
	plt.imshow(cv2.cvtColor(np.transpose(npimg, (1, 2, 0)), cv2.COLOR_BGR2RGB))
	plt.show()
