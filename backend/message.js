function msg(){
  var xhr = new XMLHttpRequest();
  var url = "apply";
  xhr.open("POST", url, true);
  xhr.setRequestHeader("Content-Type", "application/json");
  xhr.onreadystatechange = function () {
    if (xhr.readyState === 4 && xhr.status === 200) {
        var json = JSON.parse(xhr.responseText);
        console.log(json);
        //document.getElementById("counter").value = json.counter;
        alert('Response arrived!');
    }
  };
  var data = JSON.stringify({"apply": 1});
  xhr.send(data);
}

document.getElementById("counter").value = 1;
var myVar = setInterval(myTimer, 1000);

function myTimer() {
  //document.getElementById("counter").stepUp(1);
  var xhr = new XMLHttpRequest();
  var url = "url";
  xhr.open("POST", url, true);
  xhr.setRequestHeader("Content-Type", "application/json");
  xhr.onreadystatechange = function () {
    if (xhr.readyState === 4 && xhr.status === 200) {
        var json = JSON.parse(xhr.responseText);
        console.log(json);
        document.getElementById("counter").value = json.counter;
        var triggers_str = json.triggers.split(",");
        var triggers = []
        for (var i = 0; i < 3; i++)   {
            trigger_str = triggers_str[i]
            console.log("trigger_str:" + trigger_str)
            triggers.push(Number(trigger_str))
            console.log("triggers:" + triggers)
        }
        for (var i = 0; i < 3; i++) {
            document.getElementById("triggerTable").rows[i+1].cells[1].innerHTML = triggers[i];
        }
    }
  };
  triggers_str = document.getElementById("triggerTable").rows[1].cells[1].innerHTML;
  for (var i = 2; i <= 3; i++) {
    triggers_str += ", " + document.getElementById("triggerTable").rows[i].cells[1].innerHTML;
  }
  var data = JSON.stringify({"triggers": triggers_str, "counter": document.getElementById("counter").value});
  console.log("data to send" + data)
  xhr.send(data);
}

function pauseCounter() {
    clearInterval(myVar);
}

function resumeCounter() {
    myVar = setInterval(myTimer, 1000);
}