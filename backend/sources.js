var headersObj = new Object();
{
    var headerCells = document.getElementById("sourcesTable").rows[0].children;
    for (var i = 0; i < headerCells.length; i++)  {
        var header = headerCells[i].innerHTML;
        headersObj[header] = i;
    }
}

var checkCol = headersObj["check"];
var indexCol = headersObj["ind"];

function apply_save() {
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
    table.children[0].appendChild(row);
}

function remove()	{
	var table = document.getElementById("sourcesTable");
	var rows = table.rows;
	for (var row of Array.from(rows).slice(1))	{
    	var checkBox = row.cells[checkCol].children[0];
    	if (checkBox.checked == true && rows.length > 2)	{
	    		table.children[0].removeChild(row);
    	}
    }
}