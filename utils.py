umbral = 100000

def similarity_percentage(distance):
    a = distance / umbral
    percentage = (1-a)*100
    return (round(percentage, 2))


def get_eye_path(polar_image_path):
    path = polar_image_path[13:19]
    print('bd_salida/'+path+'.png')
    return('bd_salida/'+path+'.png')
