var headersObj = new Object();
{
    var headerCells = document.getElementById("sourcesTable").rows[0].children;
    for (var i = 0; i < headerCells.length; i++)  {
        var header = headerCells[i].innerHTML;
        headersObj[header] = i;
    }
}

var checkCol = headersObj["del"];
var stationCol = headersObj["station"];
var hostCol = headersObj["host"];
var portCol = headersObj["port"];
var outPortCol = headersObj["out port"];

function apply_save() {
	genNamesPorts();
	var rows = document.getElementById("sourcesTable").rows;
	for (var row of Array.from(rows).slice(1))	{
		for (var col of [stationCol, hostCol, portCol, outPortCol])	{
			var valNode = row.cells[col].children[0];
			valNode.setAttribute("value", valNode.value);
		}
	}
    sendHTML();
}

function sendHTML() {
	var xhr = new XMLHttpRequest();
	var url = "saveSources";
	xhr.open("POST", url, true);
	xhr.setRequestHeader("Content-Type", "application/html");
	var pageHTML = "<html>\n" + document.documentElement.innerHTML + "\n</html>";
	console.log(pageHTML);
	xhr.send(pageHTML);
}

function addSource() {
    var table = document.getElementById("sourcesTable");
    var rows = table.rows;
    var len = rows.length
    var row = rows[len - 1].cloneNode(true);
    row.cells[stationCol].children[0].setAttribute("value", "");
    table.children[0].appendChild(row);
	genNamesPorts();
}

function genName(name, names)	{
	if (!name)	{
		name = "ND01";
	}
	if (names.has(name))	{
		var newName;
		for (var i = 2; i < 20; i++) {
			newName = name.slice(0, 2) + ('0' + i).slice(-2);
			if (names.has(newName) == false)	{
				name = newName;
				break;
			}
		}
	}
	return name;
}

function genPort(port, ports)	{
	if (!ports.has(port))
		return port;
	console.log('has port:' + port);
	for (var port = 10001; port < 10100; port++) {
		if (ports.has(port) == false)	{
			console.log('gen port:' + port);
			return port;
		}
	}
}

function genNamesPorts()	{
	var stations = new Set();
	var ports = new Set();
	var rows = document.getElementById("sourcesTable").rows;
	for (var row of Array.from(rows).slice(1))	{
		var cells = row.cells;
	    var station = cells[stationCol].children[0].value.trim();
		console.log('station:' + station);
	    station = genName(station, stations);
	    cells[stationCol].children[0].value = station;
	    stations.add(station);
		for (var col of [portCol, outPortCol])	{
			port = parseInt(cells[col].children[0].value);
			console.log('port:' + port);
			port = genPort(port, ports);
			cells[col].children[0].value = genPort(port, ports);
			ports.add(port);
		}
    }
}

function removeSource(row)	{
	var table = document.getElementById("sourcesTable");
	var rows = Array.from(table.rows).slice(1);
	if (rows.length > 1)	{
		table.children[0].removeChild(row);
	}
}