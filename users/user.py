from flask import Blueprint, render_template, request, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from mydb import db, models
from  mydb.model import User

# Create a blueprint for user management
bp = Blueprint('user', __name__)

# Define routes for user management
@bp.route('/users')
def users():
    users = User.query.all()
    return render_template('users.html', users=users)


@bp.route('/users/create', methods=['GET', 'POST'])
def create_user():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'

        if error is None:
            user = User.query.filter_by(username=username).first()
            if user:
                error = 'Username already exists.'
            else:
                new_user = User(username=username, password=generate_password_hash(password))
                db.session.add(new_user)
                db.session.commit()
                flash('User created successfully.')
                return redirect(url_for('user.users'))

        flash(error)

    return render_template('create_user.html')


@bp.route('/users/<int:user_id>/update', methods=['GET', 'POST'])
def update_user(user_id):
    user = User.query.get_or_404(user_id)

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'

        if error is None:
            user.username = username
            user.password = generate_password_hash(password)
            db.session.commit()
            flash('User updated successfully.')
            return redirect(url_for('user.users'))

        flash(error)

    return render_template('update_user.html', user=user)


@bp.route('/users/<int:user_id>/delete', methods=['POST'])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash('User deleted successfully.')
    return redirect(url_for('user.users'))

