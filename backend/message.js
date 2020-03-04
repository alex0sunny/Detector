var headersObj = new Object();
{
    var headerCells = document.getElementById("triggerTable").rows[0].children;
    for (var i = 0; i < headerCells.length; i++)  {
        var header = headerCells[i].innerHTML;
        headersObj[header] = i;
    }
}
//console.log('headersObj:' + JSON.stringify(headersObj));
var channelCol = headersObj["channel"];
var triggerCol = headersObj["val"];
var indexCol = headersObj["ind"];

function initPage() {
	alert("onLoad");
	var xhr = new XMLHttpRequest();
	var url = "load";
	xhr.open("POST", url, true);
	xhr.setRequestHeader("Content-Type", "application/json");
	xhr.onreadystatechange = function () {
		if (xhr.readyState === 4 && xhr.status === 200) {
		    //console.log("do nothing now")
		}
	};
	var rows = document.getElementById("triggerTable").rows;
	rows.forEach(function (row) {
	    row.cells[triggerCol].innerHTML = 0 + "";
	})
	var data = JSON.stringify({"load": 1});
	xhr.send(data);
}

function apply_save() {
    apply();
    sendHTML();
}

function apply() {
    alert('apply');
	var xhr = new XMLHttpRequest();
	var url = "apply";
	xhr.open("POST", url, true);
	xhr.setRequestHeader("Content-Type", "application/json");
	xhr.onreadystatechange = function () {
		if (xhr.readyState === 4 && xhr.status === 200) {
			var json = JSON.parse(xhr.responseText);
			//console.log(json);
			//document.getElementById("counter").value = json.counter;
			//alert('Response arrived!');
		}
	};
	var rows = document.getElementById("triggerTable").rows;
	var channels = [];
	for (var j = 1; j < rows.length; j++) {
		var row = rows[j];
		var channelCell = row.cells[channelCol];
//		console.log('inner html:' + channelCell.innerHTML);
//		console.log('channel cell child:' + channelCell.children[0].innerHTML);
//		console.log('option html:' + channelCell.children[0].options[0].innerHTML);
		var options = channelCell.children[0].options;
		var selectedIndex = options.selectedIndex;
//		console.log('selected index:' + options.selectedIndex);
		var channel = options[selectedIndex].text;
		channels.push(channel);
	};
//	console.log('channels:' + channels.toString());
	setSelectedChannels(channels);
	var data = JSON.stringify({"apply": 1, "channels":  channels.join(" ")});
	xhr.send(data);
}

function sendHTML() {
	var xhr = new XMLHttpRequest();
	var url = "save";
	xhr.open("POST", url, true);
	xhr.setRequestHeader("Content-Type", "application/html");
	var pageHTML = "<html>\n" + document.documentElement.innerHTML + "\n</html>";
	xhr.send(pageHTML);
}

var myVar = setInterval(myTimer, 1000);

function myTimer() {
	//document.getElementById("counter").stepUp(1);
	var xhr = new XMLHttpRequest();
	var url = "url";
	xhr.open("POST", url, true);
	xhr.setRequestHeader("Content-Type", "application/json");
	var pageMap = getFromHtml();
//	console.log('key type:' + typeof(Object.keys(pageMap.triggers)[0]) + ' val type:' +
//	            typeof(Object.values(pageMap.triggers)[0]));
	xhr.onreadystatechange = function () {
		if (xhr.readyState === 4 && xhr.status === 200) {
		    //console.log('response:' + xhr.responseText);
			var json = JSON.parse(xhr.responseText);
//			console.log('json vals:' + Object.values(json));
//			console.log('trigger keys:' + Object.keys(json.triggers));
//			console.log('trigger vals:' + Object.values(json.triggers));
			var jsonObj = getFromJson(json);
			var triggers = jsonObj.triggers;
			setTriggers(triggers);
			if ("channels" in jsonObj)    {
                var channels = jsonObj.channels;
//                console.log("page channels:" + pageMap.channels.toString() + "; channels:" + channels.toString());
                if (channels.toString() != pageMap.channels.toString()) {
                    setChannelsList(channels);
                }
            }
		}
	};
	var data = JSON.stringify(pageMap.triggers);
	xhr.send(data);
}

function pauseCounter() {
    clearInterval(myVar);
}

function resumeCounter() {
    myVar = setInterval(myTimer, 1000);
}

function getFromJson(json) {
	var jsonObj = {"triggers" : json.triggers};
	if ("channels" in json) {
		var channels = json.channels.split(" ");
    	channels.sort();
	    jsonObj["channels"] = channels;
//	    console.log("channels:" + channels);
	}
//	console.log("triggers:" + jsonObj.triggers);
	return jsonObj;
}

function getFromHtml()	{
	var rows = document.getElementById("triggerTable").rows;
	var triggers = {};
	var channels = [];
	var i;
	if (rows.length > 1) {
		for (i = 1; i < rows.length; i++)	{
		    row = rows[i];
		    ind = parseInt(row.cells[indexCol].innerHTML);
//		    console.log('ind type:' + typeof(ind));
			triggers[ind] = parseInt(row.cells[triggerCol].innerHTML);
//			console.log('trigger val type:' + typeof(triggers[ind]));
		}
		var options = rows[1].cells[channelCol].children[0].options;
		for (i = 0; i < options.length; i++) {
		    optionNode = options[i];
		    channels.push(optionNode.text);
		}
		channels.sort();
	}
	return {triggers : triggers, channels : channels};
}

function getSelectedChannels () {
	var rows = document.getElementById("triggerTable").rows;
	var selectedChannels = [];
	if (rows.length > 1) {
		for (var i = 1;  i < rows.length; i++) {
			options = rows[i].cells[channelCol].children[0].children;
			var selectedIndex = options.selectedIndex;
			var selectedChannel = options[selectedIndex].text;
			selectedChannels.push(selectedChannel);
		}
	}
	return selectedChannels;
}

function setTriggers(triggers) {
	var rows = document.getElementById("triggerTable").rows;
	if (rows.length > 1) {
		for (var i = 1;  i < rows.length; i++) {
		    row = rows[i];
		    ind = row.cells[indexCol].innerHTML;
		    //console.log('ind:' + ind + ' triggers keys:' + Object.keys(triggers));
		    if (ind in triggers) {
			    row.cells[triggerCol].innerHTML = triggers[ind];
			}
		}
	}
}

function setChannelsList(channels) {
	var rows = document.getElementById("triggerTable").rows;
	if (rows.length > 1) {
		for (var i = 1;  i < rows.length; i++) {
			var channelCell = rows[i].cells[channelCol];
			var options = channelCell.children[0].options;
			var selectedIndex = options.selectedIndex;
//			console.log("selectedIndex:" + selectedIndex);
			var selectedChannel = options[selectedIndex].text;
			channelCell.children[0].innerHTML = "";
			channels.forEach(function (channel) {
				var optionNode = document.createElement("option");
				optionNode.text = channel;
				if (channel == selectedChannel) {
					optionNode.setAttribute("selected", "selected");
				}
				channelCell.children[0].appendChild(optionNode);
			});
		}
	}
}

function setSelectedChannels(selectedChannels) {
	var rows = document.getElementById("triggerTable").rows;
	if (rows.length > 1) {
		for (var i = 1;  i < rows.length; i++) {
			var options = rows[i].cells[channelCol].children[0].options;
			var selectedChannel = selectedChannels[i-1];
			var present = false;
			for (var j = 0; j < options.length; j++)	{
				if (options[j].text == selectedChannel) {
					options.selectedIndex = j;
					options[j].setAttribute("selected", "selected");
					present = true;
				} else {
					options[j].removeAttribute("selected");
				}
			}
			if (present == false) {
				var optionNode = document.createElement("option");
				optionNode.text = selectedChannel;
				optionNode.setAttribute("selected", "selected");
				document.getElementById("triggerTable").rows[i].cells[channelCol].children[0].appendChild(optionNode);
				selectedIndex = options.length - 1;
				options.selectedIndex = selectedIndex;
			}
		}
	}
}

function getDefaultChannels() {
	var channels = [];
	var rows = document.getElementById("triggerTable").rows;
	if (rows.length > 1) {
		for (var i = 1;  i < rows.length; i++) {
			var channelCell = rows[i].cells[channelCol];
			var options = channelCell.children[0].options;
			var channel;
			for (var j = 0; j < options.length; j++)	{
				optionNode = options[j];
				if (optionNode.hasAttribute("selected"))	{
					channel = optionNode.text;
				}
			}
			if (channel == undefined)	{
				selectedIndex = options.selectedIndex;
				channel = options[selectedIndex].text;
			}
			channels.push(channel);
		}
	}
	return channels;
}

function addTrigger() {
    var table = document.getElementById("triggerTable");
    var rows = table.rows;
    var len = rows.length
    var row = rows[len - 1].cloneNode(true);
    var ind = parseInt(row.cells[indexCol].innerHTML) + 1;
    row.cells[indexCol].innerHTML = ind;
    table.appendChild(row);
}
