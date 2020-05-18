var headersObj = new Object();
{
    var headerCells = document.getElementById("triggerTable").rows[0].children;
    for (var i = 0; i < headerCells.length; i++)  {
        var header = headerCells[i].innerHTML;
        headersObj[header] = i;
    }
}
//console.log('headersObj:' + JSON.stringify(headersObj));
var stationCol = headersObj["station"];
var channelCol = headersObj["channel"];
var valCol = headersObj["val"];
var indexCol = headersObj["ind"];
var triggerCol = headersObj["trigger"];
var sessionId = Math.floor(Math.random() * 1000000) + 1;

initPage();

var stationsData;

function initPage() {
	var xhr = new XMLHttpRequest();
	xhr.open("POST", "initTrigger", true);
	xhr.setRequestHeader("Content-Type", "application/json");
	xhr.onreadystatechange = function () {
		if (xhr.readyState === 4 && xhr.status === 200) {
			stationsData = JSON.parse(xhr.responseText);
//		    console.log("respObj:" + respObj + "\nresponseText:" + xhr.responseText +
//		    		"\nkeys:" + Object.keys(respObj) + "\n" + respObj.keys);
		    var stations = Object.keys(stationsData);
		    stations.sort();
			var rows = document.getElementById("triggerTable").rows;
			for (var row of Array.from(rows).slice(1))	{
				var channelCell = row.cells[channelCol];
				var stationCell = row.cells[stationCol];
				var station = getStation(stationCell);
				setStationsCell(stations, stationCell);
				if (station in stationsData)	{
					var channels = stationsData[station]["channels"];
					setChannelsCell(channels, channelCell);
				}
			}
		}
	};
//	rows.forEach(function (row) {
//	    row.cells[valCol].innerHTML = 0 + "";
//	})
	xhr.send();
}

function apply_save() {
    apply();
    sendHTML();
    setTimeout(nullifyVals, 3000);
    //console.log('timer started');
}

function apply() {
    //alert('apply');
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
		var valCell = row.cells[valCol];
		//valCell.innerHTML = 0;
	};
//	console.log('channels:' + channels.toString());
	setSelectedChannels(channels);
	setSelectedTriggers();
	setSelectedStations();
	var data = JSON.stringify({"apply": 1, "channels":  channels.join(" ")});
	xhr.send(data);
}

function sendHTML() {
	var xhr = new XMLHttpRequest();
	var url = "save";
	xhr.open("POST", url, true);
	xhr.setRequestHeader("Content-Type", "application/html");
	var pageHTML = "<html>\n" + document.documentElement.innerHTML + "\n</html>";
	var data = JSON.stringify({"html": pageHTML, "sessionId":  sessionId});
	xhr.send(data);
}

function nullifyVals()	{
	//console.log('time out');
	var rows = document.getElementById("triggerTable").rows;
	for (var j = 1; j < rows.length; j++) {
		var valCell = rows[j].cells[valCol];
		valCell.innerHTML = 0;
	}
}

var myVar = setInterval(myTimer, 1000);

function myTimer() {
	//document.getElementById("counter").stepUp(1);
	var xhr = new XMLHttpRequest();
	var url = "trigger";
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
			var triggers = json.triggers;
			setTriggerVals(triggers);
		}
	};
	pageMap["sessionId"] = sessionId;
	var data = JSON.stringify(pageMap);
	xhr.send(data);
}

function pauseCounter() {
    clearInterval(myVar);
}

function resumeCounter() {
    myVar = setInterval(myTimer, 1000);
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
			triggers[ind] = parseInt(row.cells[valCol].innerHTML);
//			console.log('trigger val type:' + typeof(triggers[ind]));
		}
		var options = rows[1].cells[channelCol].children[0].options;
		for (i = 0; i < options.length; i++) {
		    optionNode = options[i];
		    channels.push(optionNode.text);
		}
		channels = [...(new Set(channels))].sort();
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

function setTriggerVals(triggers) {
	var rows = document.getElementById("triggerTable").rows;
	if (rows.length > 1) {
		for (var i = 1;  i < rows.length; i++) {
		    row = rows[i];
		    ind = row.cells[indexCol].innerHTML;
		    //console.log('ind:' + ind + ' triggers keys:' + Object.keys(triggers));
		    if (ind in triggers) {
			    row.cells[valCol].innerHTML = triggers[ind];
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

function setChannelsCell(channels, channelCell)	{
	var options = channelCell.children[0].options;
	var selectedChannel = options[options.selectedIndex].text;
	var channels_cur = channels.slice();
	if (!channels_cur.includes(selectedChannel))	{
		channels_cur.push(selectedChannel);
	};
	channelCell.children[0].innerHTML = "";
	channels_cur.forEach(function (channel) {
		var optionNode = document.createElement("option");
		optionNode.text = channel;
		if (channel == selectedChannel) {
			optionNode.setAttribute("selected", "selected");
		}
		channelCell.children[0].appendChild(optionNode);
	});
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

function setSelectedTriggers() {
	var rows = document.getElementById("triggerTable").rows;
	if (rows.length > 1) {
		for (var i = 1;  i < rows.length; i++) {
			var options = rows[i].cells[triggerCol].children[0].options;
			for (var j = 0; j < options.length; j++)	{
				var option = options[j];
				if (j == options.selectedIndex) {
					option.setAttribute("selected", "selected");
				} 
				else {
					option.removeAttribute("selected");
				}
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
    table.children[0].appendChild(row);
}

function getStation(stationCell)	{
	var options = stationCell.children[0].options;
	return options[options.selectedIndex].text;
}

function setStationsCell(stations, stationCell)	{
	var selectedStation = getStation(stationCell);
	var stations_cur = stations.slice();
	if (!stations_cur.includes(selectedStation))	{
		stations_cur.push(selectedStation);
	};
	stationCell.children[0].innerHTML = "";
	stations_cur.forEach(function (station) {
		var optionNode = document.createElement("option");
		optionNode.text = station;
		if (station == selectedStation) {
			optionNode.setAttribute("selected", "selected");
		}
		stationCell.children[0].appendChild(optionNode);
	});
}

function setSelectedStations(stations, stationCell)	{
	var rows = document.getElementById("triggerTable").rows;
	for (var row of Array.from(rows).slice(1))	{
		var stationCell = row.cells[stationCol];
		var selectedStation = getStation(stationCell);
		var options = stationCell.children[0].options;
		for (var option of Array.from(options))	{
			if (option.text == selectedStation)	{
				option.setAttribute("selected", "selected");
			} else	{
				option.removeAttribute("selected");
			}
		}
	}
}

function stationChange(node)	{
	var station = node.value;
	var row = node.parentNode.parentNode;
	if (!stationsData)	{
		console.log("stations data is unavailable");
	} else	{
		var channelCell = row.cells[channelCol];
		var channels = stationsData[station]["channels"];
		setChannelsCell(channels, channelCell);
	}
}