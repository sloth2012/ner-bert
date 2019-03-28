$('#confirm-delete').on('show.bs.modal', function(e) {
  var $target = $(e.relatedTarget);
  var $modal = $(this);
  if ($target.data('href')) {
    $modal.find('.danger').attr('href', $target.data('href'));
  } else if ($target.data('onclick')) {
    $modal.find('.danger').attr('onclick', $target.data('onclick'));
  }
  $("#resourceName").text($target.data('image'));
  $modal.find('.danger').bind('click', function() {
    $modal.modal('hide');
  });
});

$(function () {
  $.notify.defaults({
    position: "right-center"
  });

  $("#criteria").on("change", function() {
    window.location = window.location.pathname + "?criteria=" + $( this ).val();
  });

  $('[data-toggle="tooltip"]').tooltip();
  $('[data-toggle="popover"]').popover();
});

var isEmpty = function(param) {
  return (undefined == param || '' == param || 'None' == param);
}

var save = function() {
  var $form = $('#bioForm');
  var form_data = $form.serialize();
  $.ajax({
    type: "POST",
    url: '/bio/update',
    data: form_data
  }).done(function(data) {
    console.log(data);
    if (data.success) {
      location.reload();
    } else {
      $.notify(data.message);
    }
  });
};

var triggerNewSearch = function() {
  var $buttons = $(event.currentTarget).parent().parent().parent();
  var $actions = $buttons.find('.actions');
  var name = $actions.data('name');
  var org = $actions.data('org');
  if (isEmpty(name) || isEmpty(org)) {
    $actions.notify("Name is empty.", "error");
    return;
  }
  var params = "n=" + name + "&o=" + org + "&force=true";
  $.get('/bio/search?' + params, function(data) {
    if (data.success) {
      $actions.notify("Added to queue, name: " + name + ", org: " + org, "success");
    } else {
      $actions.notify(data.message, "error");
    }
  });
};

var triggerGenerateFromSource = function() {
  var $buttons = $(event.currentTarget).parent().parent().parent();
  var $actions = $buttons.find('.actions');
  var id = $actions.data('id');
  var name = $actions.data('name');
  var org = $actions.data('org');
  if (isEmpty(id) || isEmpty(name) || isEmpty(org)) {
    $actions.notify("Invalid request", "error");
    return;
  }

  var url = new URI("/bio/generate");
  url.setQuery("id", id);
  url.setQuery("name", name);
  url.setQuery("org", org);
  $actions.notify("Generating, it takes several seconds", "info");
  $.get(url.toString(), function(data) {
    if (data.success) {
      $actions.notify(data.message + ", please refresh to see changes", "success");
    } else {
      $actions.notify(data.message, "info");
    }
  });
};

var triggerGenerateSource = function() {
  var $btn = $(event.currentTarget);
  var $handle = $btn
  var name = $btn.data('name');
  var org = $btn.data('org');
  if (isEmpty(name) || isEmpty(org)) {
    var $buttons = $(event.currentTarget).parent().parent().parent();
    var $actions = $buttons.find('.actions');
    name = $actions.data('name');
    org = $actions.data('org');
    $handle = $actions
  }
  if (isEmpty(name) || isEmpty(org)) {
    $handle.notify('name or org is empty');
    return;
  }

  var url = new URI("/bio/source/generate");
  url.setQuery("name", name);
  url.setQuery("org", org);
  $.get(url.toString(), function(data) {
    var type = data.success ? "success" : "error";
    $handle.notify(data.message, type);
  });
};

var extraOrgs = function() {
  var $buttons = $(event.currentTarget).parent().parent().parent();
  var $actions = $buttons.find('.actions');
  var bio = $actions.data('bio');
  $.get('/orgs', {"bio": bio}, function(data) {
    console.log(data);
    if (data.message) {
      $actions.notify(data.message);
      return;
    }
    orgs = data.orgs;
    if (0 == orgs.length) {
      $actions.notify("no orgs extracted");
      return;
    }
    var $container = $buttons.parent().find('.org-list');
    $container.empty();
    for (var i in orgs) {
      var org = orgs[i];
      $container.append("<div><span class='label label-info'>" + org + "</span></div>");
    }
  });
};

var cancelJob = function() {
  var $btn = $(event.currentTarget);
  var node = $btn.data('node');
  var id = $btn.data('id');
  if (isEmpty(node)) {
    if (false === confirm("Cancel ALL jobs on ALL nodes?!")) {
      return;
    }
  } else if (isEmpty(id)) {
    if (false === confirm("Cancel ALL jobs on THIS nodes?!")) {
      return;
    }
  }

  var url = new URI("/bots/job/cancel");
  url.setQuery("node", node);
  url.setQuery("id", id);
  $.get(url.toString(), function(data) {
    console.log(data);
    $btn.notify(data.message, {position: "left center", className: "info"});
  });
};

