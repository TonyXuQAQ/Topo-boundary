input = QgsProject.instance().mapLayersByName('PAVEMENT_EDGE')[0]


# generate grid
coord_system = QgsCoordinateReferenceSystem('EPSG:2263')

# TODO: change ranges of i and j based on image tiles. And you need to map i,j to the index of image tiles.
for i in range(5): # You need to change range of i
    for j in range(10): # You need to change range of j

        course_xmin_round = 910000 + 2500 * i
        course_xmax_round = 910000 + 2500 * (i+1)
        course_ymin_round = 115000 + 2500 * i
        course_ymax_round = 115000 + 2500 * (i+1)

        course_extent = "%f,%f,%f,%f" %(course_xmin_round, course_xmax_round, course_ymin_round, course_ymax_round) 

        params = {'TYPE':2,'EXTENT': course_extent,
            'HSPACING':2500,
            'VSPACING':2500,
            'HOVERLAY':0,
            'VOVERLAY':0,
            'CRS':coord_system,
            'OUTPUT':QgsProcessing.TEMPORARY_OUTPUT}

        coarse_grid = processing.run("native:creategrid", params)['OUTPUT']
        #QgsProject.instance().addMapLayers([coarse_grid])


        # get intersection
        params = {
             'INPUT': input,
             'OVERLAY': coarse_grid,
             'OUTPUT': 'TEMPORARY_OUTPUT'
        }
                
        intersectLayer = processing.run("qgis:intersection", params)['OUTPUT']
        #QgsProject.instance().addMapLayers([intersectLayer])

        # get vertex
        params = {
             'INPUT': intersectLayer,
             'OUTPUT': 'TEMPORARY_OUTPUT'
        }
        vertices = processing.run("qgis:extractvertices",params)['OUTPUT']
        #QgsProject.instance().addMapLayers([vertices])
        
        # output vertices as csv
        QgsVectorFileWriter.writeAsVectorFormat(vertices, str(i)+'_'+str(j)+'.csv', "utf-8", vertices.crs(), "CSV", layerOptions =['GEOMETRY=AS_XY'])
        

