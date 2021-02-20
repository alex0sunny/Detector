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

function apply_save() {
	genNames();
	var rows = document.getElementById("sourcesTable").rows;
	for (var row of Array.from(rows).slice(1))	{
		for (var col of [stationCol, hostCol, portCol])	{
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

function genNames()	{
	var stations = new Set();
	var rows = document.getElementById("sourcesTable").rows;
	for (var row of Array.from(rows).slice(1))	{
		var cells = row.cells;
	    var station = cells[stationCol].children[0].getAttribute("value").trim();
	    station = genName(station, stations);
	    cells[stationCol].children[0].value = station;
	    stations.add(station);
    }
}

function removeSource(row)	{
	var table = document.getElementById("sourcesTable");
	var rows = Array.from(table.rows).slice(1);
	if (rows.length > 1)	{
		table.children[0].removeChild(row);
	}
}