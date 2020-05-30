umbral = 100000

# Calcula el porcentaje de similitud del ojo de entrada con respecto a la salida
def similarity_percentage(distance):
    a = distance / umbral
    percentage = (1-a)*100
    return (round(percentage, 2))

# Obtiene la ruta de la imagen de salida
def get_eye_path(polar_image_path):
    path = polar_image_path[13:19]
    return('bd_salida/'+path+'.png')
