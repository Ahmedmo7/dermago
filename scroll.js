$("nav ul li a").click(function(){
    document.getElementById('nav').click();
});


$(document).ready(function() {
    var scrollLink = $('.scroll');


    // Smooth scrolling
    scrollLink.click(function(e) {
        e.preventDefault();
        $('body,html').animate({
            scrollTop: $(this.hash).offset().top
        }, 1000  )
    })


})