var ruleIdCol = 0;
var formulaCol = 1;
var ruleValCol = 2;

var timerVar = setInterval(updateFunc, 1000);

var triggersObj = {"1": 1, "2": 0, "4": 0};

initFunc();

function updateTriggers (triggersObj, ruleCell) {
	var children = ruleCell.children;
	for (var i = 0; i < children.length; i++)	{
		var node = children[i];
		//console.log('nodeName:' + node.nodeName);
		if (node.nodeName == "OUTPUT") 	{
			var triggerNode = children[i - 1];
			var options = triggerNode.options;
			var selectedIndex = options.selectedIndex;
			var triggerIdStr = options[selectedIndex].text;
			//console.log('triggerIdStr:' + triggerIdStr);
			var triggerVal = triggersObj[triggerIdStr];
			if (triggerVal == undefined)	{
				triggerVal = "-";
			} 
			//console.log('triggerVal:' + triggerVal);
			node.value = triggerVal;
		} 
	}
}

function updateFunc () {
	var xhr = new XMLHttpRequest();
	xhr.open("POST", "rule", true);
	xhr.setRequestHeader("Content-Type", "application/json");
	xhr.onreadystatechange = function () {
		if (xhr.readyState === 4 && xhr.status === 200) {
		    //console.log('response:' + xhr.responseText);
			triggersObj = JSON.parse(xhr.responseText);
			var rows = document.getElementById("rulesTable").rows;
			for (var i = 0; i < rows.length; i = i + 1)	{
				var row = rows[i];
				var ruleCell = row.cells[1];
				updateTriggers(triggersObj, ruleCell);
			}
		}
	};
	xhr.send(JSON.stringify(triggersObj));
}

function initFunc () {
	var xhr = new XMLHttpRequest();
	xhr.open("POST", "initRule", true);
	xhr.setRequestHeader("Content-Type", "application/json");
	xhr.onreadystatechange = function () {
		if (xhr.readyState === 4 && xhr.status === 200) {
		    //console.log('response:' + xhr.responseText);
			var triggersIds = JSON.parse(xhr.responseText);
			var triggerId;
			for (triggerId of triggersIds)	{
				triggersObj[triggerId] = 0;
			}
		}
	}
	xhr.send();
}
