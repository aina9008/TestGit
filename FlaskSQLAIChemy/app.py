from flask import Flask,request,flash,render_template,redirect,url_for

app = Flask('Flask Framework')

app.secret_key='aina_flask_framework'

@app.route('/',methods=['GET','POST'])
def submit():
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
