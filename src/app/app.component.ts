import { Component } from '@angular/core';
import {WebcamImage, WebcamInitError} from 'ngx-webcam';
import {Subject, Observable} from 'rxjs';
declare var $: any;
import * as RecordRTC from 'recordrtc';
import { DomSanitizer } from '@angular/platform-browser';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'emotionDetectorUI';


  public webcamImage: WebcamImage = null;
  public trigger: Subject<void> = new Subject<void>();
  
  triggerSnapshot(): void {
   this.trigger.next();
  }

  handleImage(webcamImage: WebcamImage): void {
    console.info('Saved webcam image', webcamImage);
    this.webcamImage = webcamImage;
   }
    
   public get triggerObservable(): Observable<void> {
    return this.trigger.asObservable();
   }

   public handleInitError(error: WebcamInitError): void {
    if (error.mediaStreamError && error.mediaStreamError.name === "NotAllowedError") {
      console.warn("Camera access was not allowed by user!");
    }
  }

//Lets declare Record OBJ
record;
//Will use this flag for toggeling recording
recording = false;
//URL of Blob
url;
error;
constructor(private domSanitizer: DomSanitizer) {}
sanitize(url: string) {
return this.domSanitizer.bypassSecurityTrustUrl(url);
}
/**
* Start recording.
*/
initiateRecording() {
this.recording = true;
let mediaConstraints = {
video: false,
audio: true
};
navigator.mediaDevices.getUserMedia(mediaConstraints).then(this.successCallback.bind(this), this.errorCallback.bind(this));
}
/**
* Will be called automatically.
*/
successCallback(stream) {
var options = {
mimeType: "audio/wav",
numberOfAudioChannels: 2
};
//Start Actuall Recording
var StereoAudioRecorder = RecordRTC.StereoAudioRecorder;
this.record = new StereoAudioRecorder(stream, options);
this.record.record();
}
/**
* Stop recording.
*/
stopRecording() {
this.recording = false;
this.record.stop(this.processRecording.bind(this));
}
/**
* processRecording Do what ever you want with blob
* @param  {any} blob Blog
*/
processRecording(blob) {
this.url = URL.createObjectURL(blob);
console.log("blob", blob);
console.log("url", this.url);
}
/**
* Process Error.
*/
errorCallback(error) {
this.error = 'Can not play audio in your browser';
}


}
