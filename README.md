# nailTask
Classify good and bad nails

to run the docker image call
docker build -t keras-app .
docker run -t keras-app -d
curl -X POST -F image=@image-name.jpeg 'http://localhost:5000/predict'
