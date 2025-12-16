document.addEventListener('DOMContentLoaded', function() {
    const loginTab = document.getElementById('login-tab');
    const signupTab = document.getElementById('signup-tab');
    const loginForm = document.getElementById('login-form');
    const signupForm = document.getElementById('signup-form');

    // Function to switch to the Login form
    loginTab.addEventListener('click', function() {
        loginTab.classList.add('active-tab');
        signupTab.classList.remove('active-tab');
        loginForm.classList.add('active-form');
        signupForm.classList.remove('active-form');
    });

    // Function to switch to the Sign Up form
    signupTab.addEventListener('click', function() {
        signupTab.classList.add('active-tab');
        loginTab.classList.remove('active-tab');
        signupForm.classList.add('active-form');
        loginForm.classList.remove('active-form');
    });

    // NOTE on Redirection Logic:
    // The actual redirection after a successful login (e.g., to rider_view.html, analyst_view.html, etc.) 
    // requires server-side authentication (e.g., Node.js, Python, PHP). 
    // You would integrate client-side JS with your backend to handle that redirection.
    // For now, these buttons are placeholders.
});