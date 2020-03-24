from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import sys,os
sys.path.append(os.getcwd())
import bert
import numpy as np

import tensorflow as tf
from sent_processor import SentProcessor, convert_examples_to_features
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures



import bertfopredict3 as bp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

bootstrap = Bootstrap(app)
moment = Moment(app)

model = bp.model
model.summary()


class NameForm(FlaskForm):
    name = StringField('输入评论', validators=[DataRequired()])
    submit = SubmitField('Submit')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    name = None
    form = NameForm()
    label = None
    if form.validate_on_submit():
        name = form.name.data
        print(name)
        example = InputExample(guid="dummy-id", text_a=name, text_b=None, label=None)
        fea = bp.convert_single_example("dummy-id", example, bp.sp.get_labels(), bp.max_seq_len, bp.tokenizer)
        test_ids=[fea.input_ids]
        test_ids=np.array(test_ids)
        preds=model.predict(test_ids, batch_size=64)
        pred_idx=np.argmax(preds, axis=1)[0]
        labels = bp.sp.get_labels()
        print(pred_idx)
        print(labels[pred_idx])
        label = labels[pred_idx]
    return render_template('index.html', form=form, name=name, label=label)
