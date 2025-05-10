from flask import  Blueprint,request
from api_service import service_remove_background

api_v1 = Blueprint('api_v1', __name__, )

@api_v1.route('/remove_image_background', methods=['POST'])
def remove_image_background():
    return service_remove_background(request)



