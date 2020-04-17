var ruleIdCol = 0;
var formulaCol = 1;
var ruleValCol = 2;

var timerVar = setInterval(updateFunc, 1000);

var triggersObj = {"1": 0, "2": 0, "4": 0};

var rulesObj = getRulesObj();

var sessionId = Math.floor(Math.random() * 1000000) + 1;

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

function updateRules(rulesObj)	{
    var table = document.getElementById("rulesTable");
    var rows = table.rows;
    for (var i = 1; i < rows.length; i++)	{
    	var row = rows[i];
      	var ruleId = row.cells[ruleIdCol].innerHTML;
       	console.log('ruleId:' + ruleId + ' ruleVal:' + rulesObj[ruleId]);
       	row.cells[ruleValCol].innerHTML = rulesObj[ruleId];
    }	
}

function updateFunc () {
	var xhr = new XMLHttpRequest();
	xhr.open("POST", "rule", true);
	xhr.setRequestHeader("Content-Type", "application/json");
	xhr.onreadystatechange = function () {
		if (xhr.readyState === 4 && xhr.status === 200) {
		    //console.log('response:' + xhr.responseText);
			respObj = JSON.parse(xhr.responseText);
			triggersObj = respObj['triggers'];
			rulesObj = respObj['rules'];
			console.log('rulesObj from server:' + JSON.stringify(rulesObj));
			updateRules(rulesObj);
			var rows = document.getElementById("rulesTable").rows;
			for (var i = 1; i < rows.length; i = i + 1)	{
				var row = rows[i];
				var ruleCell = row.cells[formulaCol];
				updateTriggers(triggersObj, ruleCell);
			}
		}
	};
	var rulesObj = getRulesObj();
	console.log('rulesObj:' + JSON.stringify(rulesObj));
	var data = {triggers: triggersObj, sessionId: sessionId, rules: rulesObj};
	xhr.send(JSON.stringify(data));
}

function initFunc () {
	var xhr = new XMLHttpRequest();
	xhr.open("POST", "initRule", true);
	xhr.setRequestHeader("Content-Type", "application/json");
	xhr.onreadystatechange = function () {
		if (xhr.readyState === 4 && xhr.status === 200) {
		    //console.log('response:' + xhr.responseText);
			var triggersIds = JSON.parse(xhr.responseText);
			console.log('triggersIds:' + triggersIds);
			for (var triggerId of triggersIds)	{
				triggersObj[triggerId] = 0;
			}
			var rows = document.getElementById("rulesTable").rows;
			for (var i = 1; i < rows.length; i = i + 1)	{
				var row = rows[i];
				var ruleCell = row.cells[formulaCol];
				fillTriggers(triggersIds, ruleCell);
			}
		}
	}
	console.log('triggers obj:' + triggersObj);
	xhr.send();
}

function fillTriggers (triggersIds, ruleCell)	{
	var children = ruleCell.children;
	for (var i = 0; i < children.length; i++)	{
		var node = children[i];
		if (node.nodeName == "OUTPUT") 	{
			var triggerNode = children[i - 1];
			var options = triggerNode.options;
			var selectedIndex = options.selectedIndex;
			var selectedTrigger = options[selectedIndex].text;
			var option = options[0];
			if (selectedIndex == 0)	{
				option.setAttribute("selected", "selected");
			} else	{
				option.removeAttribute("selected");
			}
			triggerNode.innerHTML = "";
			triggerNode.appendChild(option);
			triggersIds.forEach(function (trigger) {
				option = document.createElement("option");
				option.text = trigger;
				if (trigger == selectedTrigger) {
					option.setAttribute("selected", "selected");
				}
				triggerNode.appendChild(option);
			});
		} 
	}	
}

function addRule() {
    var table = document.getElementById("rulesTable");
    var rows = table.rows;
    var len = rows.length
    var row = rows[len - 1].cloneNode(true);
    var ind = parseInt(row.cells[ruleIdCol].innerHTML) + 1;
    row.cells[ruleIdCol].innerHTML = ind;
    table.children[0].appendChild(row);
}

function setSelected(node) 	{
	if (node.nodeName == "select")	{
		var options = node.options;
		var selectedIndex = options.selectedIndex;
		for (var i = 0; i < options.lengh; i++)	{
			var option = options[i];
			if (i == selectedIndex)	{
				option.setAttribute("selected", "selected");
			} else	{
				option.removeAttribute("selected");
			}
		}
	}	
}

function apply()	{
    var table = document.getElementById("rulesTable");
    var rows = table.rows;
    for (var i = 1; i < rows.length; i++)	{
    	var row = rows[i];
    	var ruleCell = row.cells[formulaCol];
    	var children = ruleCell.children;
    	for (var j = 0; j < children.length; j++)	{
    		var node = children[j];
    		setSelected(node);
    	}
    }
	var xhr = new XMLHttpRequest();
	var url = "applyRules";
	xhr.open("POST", url, true);
	xhr.setRequestHeader("Content-Type", "application/html");
	var pageHTML = "<html>\n" + document.documentElement.innerHTML + "\n</html>";
	xhr.send(pageHTML);
}

function getRulesObj()	{
	var rulesObj = {};
    var table = document.getElementById("rulesTable");
    var rows = table.rows;
    for (var i = 1; i < rows.length; i++)	{
    	var row = rows[i];
    	var ruleId = parseInt(row.cells[ruleIdCol].innerHTML);
    	var ruleVal = parseInt(row.cells[ruleValCol].innerHTML);
    	rulesObj[ruleId] = ruleVal;
    }
    return rulesObj;
}