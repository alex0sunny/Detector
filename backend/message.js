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
//        console.log(json);
//        document.getElementById("counter").value = json.counter;
        var triggers_str = json.triggers.split(",");
        var triggers = []
        var channels = ["ch1", "ch2", "ch3"]
        for (var i = 0; i < 3; i++)   {
            trigger_str = triggers_str[i]
            //console.log("trigger_str:" + trigger_str)
            triggers.push(Number(trigger_str))
            //console.log("triggers:" + triggers)
        }
        for (var i = 1; i <= 3; i++) {
            var channelCell = document.getElementById("triggerTable").rows[i].cells[1];
            console.log("current channel cell:" + channelCell.innerHtml);
            console.log("row:" + document.getElementById("triggerTable").rows[i].innerHtml);
            var optionNodes = Array.prototype.slice.call(channelCell.childNodes[0].childNodes);
            var currentChannels = [];
            optionNodes.forEach(function(optionNode){
                currentChannels.push(optionNode.getAttribute("value"));
            });
            channels.sort();
            currentChannels.sort();
            console.log("current channels:" + currentChannels + " sample channels:" + channels);
            if (channels.toString() != currentChannels.toString()) {
                console.log("channels sets are different!");
                channelCell.innerHTML = "<select>" +
                                        "<option value=\"ch1\">CHE</option>" +
                                        "<option value=\"ch2\">CHN</option>" +
                                        "<option value=\"ch3\">CHZ</option>" +
                                        "</select>";
            }
        }
        for (var i = 0; i < 3; i++) {
            document.getElementById("triggerTable").rows[i+1].cells[2].innerHTML = triggers[i];
        }
        console.log("channel:" + document.getElementById("triggerTable").rows[1].cells[1].innerHTML);
    }
  };
  triggers_str = document.getElementById("triggerTable").rows[1].cells[2].innerHTML;
  for (var i = 2; i <= 3; i++) {
    triggers_str += ", " + document.getElementById("triggerTable").rows[i].cells[2].innerHTML;
  }
  var data = JSON.stringify({"triggers": triggers_str, "counter": document.getElementById("counter").value});
  //console.log("data to send" + data)
  xhr.send(data);
}

function pauseCounter() {
    clearInterval(myVar);
}

function resumeCounter() {
    myVar = setInterval(myTimer, 1000);
}