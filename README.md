# QR Recognitation

## Ex2
<h6>submitters:</h6>
<p>Name: liav levi, ID:206603193</p>
<p>Name: Barak Sharabi, ID:</p>
<p>Name: Oria Zadok, ID:315500157</p>
<p>Name: Sagi Yosef Azulai, ID:207544230</p>

## HOW TO RUN
<p> first thing to do for this assignment running and checking: </p>


~~~
pip install -r requirements.txt 
~~~

<p> if you are using windows you may need to install with --user flag like this:</p>

~~~
pip install -r requirements.txt --user
~~~

<p> to run the assignment</p>

~~~
python main.py <video_path>
~~~

<p> to run the tests(we disable warning as we use old version of pandas and numpy)</p>

~~~
pytest --disable-warnings
python -m Tests.TestDroneSimulatorTest


~~~