package LibraVDB::Database;
use strict;
use warnings;
use LibraVDB qw( :all );
use LibraVDB::Collection;

sub new {
    my ($class, $path) = @_;
    
    my $db_id = LibraVDB::OpenDB($path);
    if ($db_id < 0) {
        die "Failed to open database at path: $path\n";
    }
    
    my $self = {
        db_id => $db_id,
        path  => $path,
    };
    bless $self, $class;
    return $self;
}

sub create_collection {
    my ($self, $name, $dimension) = @_;
    
    my $col_id = LibraVDB::CreateCollection($self->{db_id}, $name, $dimension);
    if ($col_id < 0) {
        die "Failed to create collection: $name\n";
    }
    
    return LibraVDB::Collection->new($self->{db_id}, $col_id, $name, $dimension);
}

sub get_collection {
    my ($self, $name, $dimension) = @_;
    
    my $col_id = LibraVDB::GetCollection($self->{db_id}, $name);
    if ($col_id < 0) {
        die "Failed to get collection: $name\n";
    }
    
    return LibraVDB::Collection->new($self->{db_id}, $col_id, $name, $dimension);
}

sub vacuum {
    my ($self) = @_;
    my $err = LibraVDB::Vacuum($self->{db_id});
    _check_error($err);
}

sub drop_database {
    my ($self) = @_;
    my $err = LibraVDB::DropDatabase($self->{db_id});
    _check_error($err);
}

sub _check_error {
    my ($err_str) = @_;
    return unless defined $err_str;
    return if $err_str eq 'OK';
    
    if ($err_str =~ /^ERROR:/ || $err_str =~ /^error/i) {
        die "$err_str\n";
    }
}

sub query {
    my ($self, $sql) = @_;
    my $res_str = LibraVDB::DatabaseQuery($self->{db_id}, $sql);
    return {} unless defined $res_str;
    if ($res_str =~ /^{"error"/) {
        require JSON::PP;
        my $err = JSON::PP::decode_json($res_str);
        die "Query failed: " . $err->{error} . "\n";
    }
    require JSON::PP;
    return JSON::PP::decode_json($res_str);
}

sub query_with_params {
    my ($self, $sql, $params) = @_;
    require JSON::PP;
    my $params_str = defined $params ? JSON::PP::encode_json($params) : "";
    my $res_str = LibraVDB::DatabaseQueryWithParams($self->{db_id}, $sql, $params_str);
    return {} unless defined $res_str;
    if ($res_str =~ /^{"error"/) {
        my $err = JSON::PP::decode_json($res_str);
        die "QueryWithParams failed: " . $err->{error} . "\n";
    }
    return JSON::PP::decode_json($res_str);
}

sub DESTROY {
    my ($self) = @_;
    if (defined $self->{db_id} && $self->{db_id} >= 0) {
        LibraVDB::CloseDB($self->{db_id});
        $self->{db_id} = -1;
    }
}

1;
