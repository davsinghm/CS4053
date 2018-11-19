# Traffic Light Detection

## Compile
```
sudo apt install libopencv-dev pkg-config
g++ detect_traffic_light.cpp -o detectlight `pkg-config opencv --cflags` `pkg-config opencv --libs`
```

## Run
Run the traffic light detection for a given sample file in CamVidLights/CamVidLights<file_no>.png
```
./detectlight <file_no>
```

Screenshots
![Alt text](screenshots/screenshot02.png?raw=true "Screenshot 1" )
![Alt text](screenshots/screenshot09.png?raw=true "Screenshot 2" )
