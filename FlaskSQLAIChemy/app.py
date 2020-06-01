from flask import Flask,request,flash,render_template,redirect,url_for
from flask_sqlalchemy import SQLAlchemy
#from Flask_Framework import train

app = Flask('Flask Framework')
# db=SQLAlchemy(app)
#
# app.config['SQLALCHEMY_DATABASE_URI']='sqlserver:'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
app.secret_key='aina_flask_framework'

@app.route('/',methods=['GET','POST'])
def submit():
    #x, y, x_test, y_test=train.loadData(True)
    #train.trainRegression(x, y, x_test, y_test)
    #train.trainNN(x, y, x_test, y_test)
    #train.trainCNN(x, y, x_test, y_test)
    if request.method=='POST':
        username=request.form.get('username')
        password=request.form.get('password')
        if not all([username,
                   password]):
            flash('user name or password not empty')
        else:
            return redirect(url_for('login'))

    return  render_template('login.html')

@app.route('/login')
def login():
    return 'Login successed.'

if __name__ == '__main__':
    app.run()
