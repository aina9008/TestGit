from flask import Flask

app = Flask('Flask Framework')


@app.route('/',methods=['GET','POST'])
def submit():
    return  'Hello my flask!'

@app.route('/login')
def login():
    return 'Login successed.'

if __name__ == '__main__':
    app.run()
