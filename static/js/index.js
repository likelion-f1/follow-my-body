
const FPS = 30; // similar FPS with video
const r_FPS = 1; // analyze FPS
var W = 640, H = 360;

var socket = io('http://localhost:5000');
var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
var video = document.getElementById('video');
var frame_cnt = 0;

//video.width = W;
//video.height = H; ;

socket.on('connect', function(){
    console.log("Connected...!", socket.connected)
});

function showImage(image) {
    const image_id = document.getElementById('image');
    image_id.src = image;
}

socket.on('response_back', showImage);


// video streaming
if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
        video.srcObject = stream;
        video.play();
    })
    .catch(function (err0r) {
        console.log(err0r);
        console.log("Something went wrong!");
    });
}

// Trigger photo take
document.getElementById("snap").addEventListener("click", function() {
    context.drawImage(video, 0, 0, W, H);
    var request = new XMLHttpRequest();
    request.open('POST', '/submit?image=' + video.toString('base64'), true);
    request.send();
});

cv['onRuntimeInitialized']=()=>{

    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
    let cap = new cv.VideoCapture(video);

    function run() {
         if (!socket.connected) {
            console.log('socket not connected!');
            return;
        }

        frame_cnt ++;
//        cap.read(src);
        context.drawImage(video, 0, 0, W, H);

        if (frame_cnt % parseInt(FPS / r_FPS) == 0) {
            console.log('frame cnt:' + frame_cnt);
            var type = "image/png";
            var data = canvas.toDataURL(type);
            data = data.replace('data:' + type + ';base64,', ''); //split off junk at the beginning
            socket.emit('analyze', data);
        }
    }

    // run analyze
    setInterval(run, 1000/FPS);

}

// Check camera stream is playing by getting its width
video.addEventListener('playing', function() {
    if (this.videoWidth === 0) {
        console.error('videoWidth is 0. Camera not connected?');
    } else {
        console.log('video W, H :' + this.videoWidth + ', ' + this.videoHeight)
        W = this.videoWidth
        H = this.videoHeight
    }
}, false);



/*
var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
var video = document.getElementById('video');

// Trigger photo take
document.getElementById("snap").addEventListener("click", function() {
    context.drawImage(video, 0, 0, 640, 480);
var request = new XMLHttpRequest();
request.open('POST', '/submit?image=' + video.toString('base64'), true);
request.send();
});
*/
