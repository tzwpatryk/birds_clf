from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired

class ImageForm(FlaskForm):
    image = FileField('Image', validators=[DataRequired()])
    submit = SubmitField('Predict')